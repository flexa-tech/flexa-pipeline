"""
Option 3: HaMeR 2D wrist detection + R3D LiDAR depth → 3D hand positions → H1 replay.

This bypasses HaMeR's weak depth estimation by using actual LiDAR measurements.
"""
import json, numpy as np, zipfile, os, mujoco, cv2
import tempfile, shutil, subprocess
from pathlib import Path
from scipy.ndimage import uniform_filter1d
import liblzfse

from replay_egodex_on_h1 import (
    MuJoCoArmIK, ARM_JOINTS, FINGER_CURL_JOINTS, fingertip_distance_to_curl
)

ASSETS_DIR = Path(__file__).parent / "humanoidbench_assets"
WIDTH, HEIGHT = 480, 360
HAMER_FOCAL = 5000.0

# ===== Step 1: Load R3D data =====
print("Loading R3D data...")
r3d_path = 'raw_captures/stack2/stack2.r3d'
z = zipfile.ZipFile(r3d_path, 'r')
meta = json.loads(z.read('metadata'))

K = np.array(meta['K']).reshape(3, 3)  # Column-major intrinsic matrix
# R3D K is transposed: K[0][0]=fx, K[1][1]=fy, K[2][0]=cx, K[2][1]=cy
fx = K[0, 0]
fy = K[1, 1]
cx = K[2, 0]
cy = K[2, 1]
img_w = meta['w']  # 960
img_h = meta['h']  # 720
dw = meta['dw']    # 256
dh = meta['dh']    # 192

print(f"Camera: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
print(f"Image: {img_w}x{img_h}, Depth: {dw}x{dh}")

# Parse poses (7 values per frame: qx, qy, qz, qw, tx, ty, tz)
poses = np.array(meta['poses'])  # Already (N, 7) list of lists
n_poses = len(poses)
print(f"Poses: {poses.shape} = {n_poses} frames")

# Get depth file list
depth_files = sorted([n for n in z.namelist() if n.endswith('.depth')])
rgb_files = sorted([n for n in z.namelist() if n.endswith('.jpg')])
print(f"Depth files: {len(depth_files)}, RGB files: {len(rgb_files)}")

def load_depth(frame_idx):
    """Load and decompress a depth map."""
    if frame_idx >= len(depth_files):
        return None
    compressed = z.read(depth_files[frame_idx])
    dec = liblzfse.decompress(compressed)
    return np.frombuffer(dec, dtype=np.float32).reshape(dh, dw)

def quat_to_rotmat(q):
    """Convert quaternion [qx, qy, qz, qw] to 3x3 rotation matrix."""
    qx, qy, qz, qw = q
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
        [2*(qx*qy+qw*qz), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qw*qx)],
        [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx*qx+qy*qy)]
    ])

def get_camera_transform(frame_idx):
    """Get 4x4 camera transform for a frame."""
    # Interpolate if frame_idx is between pose keyframes
    pose_idx = min(frame_idx * n_poses // len(depth_files), n_poses - 1)
    pose = poses[pose_idx]
    R = quat_to_rotmat(pose[:4])
    t = pose[4:7]
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

# ===== Step 2: Load HaMeR results =====
print("\nLoading HaMeR results...")
with open('modal_results/stack2_hamer_bimanual.json') as f:
    hamer_data = json.load(f)
hamer_results = hamer_data['results']
n_hamer = len(hamer_results)
print(f"HaMeR: {n_hamer} frames")

# ===== Step 3: Combine HaMeR 2D + LiDAR depth =====
print("\nCombining HaMeR 2D + LiDAR depth...")

wrist_3d = {'left': [], 'right': []}

for fi in range(n_hamer):
    frame = hamer_results[fi]
    frame_idx = frame['frame_idx']
    
    # Load depth map
    depth_map = load_depth(frame_idx)
    if depth_map is None:
        for side in ['left', 'right']:
            wrist_3d[side].append([np.nan]*3)
        continue
    
    cam_T = get_camera_transform(frame_idx)
    
    for hand in frame['hands']:
        # Determine side
        is_right = hand.get('hand', 0) == 1
        if 'bbox_center_x' in hand:
            is_right = hand['bbox_center_x'] < img_w / 2  # Left side of egocentric = right hand
        side = 'right' if is_right else 'left'
        
        bbox = hand['bbox']  # [x1, y1, x2, y2]
        x1, y1, x2, y2 = bbox
        bbox_cx = (x1 + x2) / 2
        bbox_cy = (y1 + y2) / 2
        box_size = max(x2 - x1, y2 - y1)
        
        # Use HaMeR's cam_t to get a more precise 2D wrist position within the bbox
        s, cam_tx, cam_ty = hand['cam_t']
        kps = np.array(hand['keypoints_3d'])
        
        # In the crop: wrist_crop_normalized = s * kps[0,:2] + [cam_tx, cam_ty]
        # This is in [-1, 1] normalized coordinates within the crop
        wrist_norm_x = s * kps[0, 0] + cam_tx
        wrist_norm_y = s * kps[0, 1] + cam_ty
        
        # Convert from normalized [-1,1] to pixel within crop, then to full image
        wrist_crop_x = (wrist_norm_x + 1) / 2 * box_size
        wrist_crop_y = (wrist_norm_y + 1) / 2 * box_size
        wrist_px = wrist_crop_x + (bbox_cx - box_size / 2)
        wrist_py = wrist_crop_y + (bbox_cy - box_size / 2)
        
        # Clamp to image bounds
        wrist_px = np.clip(wrist_px, 5, img_w - 5)
        wrist_py = np.clip(wrist_py, 5, img_h - 5)
        
        # Clamp to image bounds
        wrist_px = np.clip(wrist_px, 0, img_w - 1)
        wrist_py = np.clip(wrist_py, 0, img_h - 1)
        
        # Map to depth map coordinates
        depth_x = int(wrist_px * dw / img_w)
        depth_y = int(wrist_py * dh / img_h)
        depth_x = np.clip(depth_x, 0, dw - 1)
        depth_y = np.clip(depth_y, 0, dh - 1)
        
        # Read LiDAR depth (with 3x3 neighborhood average for robustness)
        y_lo = max(0, depth_y - 1)
        y_hi = min(dh, depth_y + 2)
        x_lo = max(0, depth_x - 1)
        x_hi = min(dw, depth_x + 2)
        neighborhood = depth_map[y_lo:y_hi, x_lo:x_hi]
        valid_depths = neighborhood[neighborhood > 0]
        
        if len(valid_depths) > 0:
            lidar_depth = np.median(valid_depths)
        else:
            lidar_depth = np.nan
        
        # Project 2D + depth → 3D camera frame
        X_cam = (wrist_px - cx) * lidar_depth / fx
        Y_cam = (wrist_py - cy) * lidar_depth / fy
        Z_cam = lidar_depth
        
        # Transform to world frame using R3D camera pose
        pos_cam = np.array([X_cam, Y_cam, Z_cam])
        pos_world = cam_T[:3, :3] @ pos_cam + cam_T[:3, 3]
        
        wrist_3d[side].append(pos_world.tolist())
    
    # Fill missing sides
    for side in ['left', 'right']:
        if len(wrist_3d[side]) <= fi:
            wrist_3d[side].append([np.nan]*3)

for side in ['left', 'right']:
    wrist_3d[side] = np.array(wrist_3d[side])

print(f"Right wrist 3D: {wrist_3d['right'].shape}")
print(f"Left wrist 3D: {wrist_3d['left'].shape}")

# Check ranges
for side in ['right', 'left']:
    valid = ~np.isnan(wrist_3d[side][:, 0])
    if valid.sum() > 0:
        pos = wrist_3d[side][valid]
        travel = np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1))
        print(f"\n{side} wrist (world frame):")
        print(f"  X: [{pos[:,0].min():.3f}, {pos[:,0].max():.3f}] range={pos[:,0].max()-pos[:,0].min():.3f}")
        print(f"  Y: [{pos[:,1].min():.3f}, {pos[:,1].max():.3f}] range={pos[:,1].max()-pos[:,1].min():.3f}")
        print(f"  Z: [{pos[:,2].min():.3f}, {pos[:,2].max():.3f}] range={pos[:,2].max()-pos[:,2].min():.3f}")
        print(f"  Total travel: {travel:.3f}m")
        print(f"  Valid frames: {valid.sum()}/{len(wrist_3d[side])}")

# ===== Step 4: Smooth and prepare for replay =====
for side in ['left', 'right']:
    # Interpolate NaN
    for ax in range(3):
        vals = wrist_3d[side][:, ax]
        nans = np.isnan(vals)
        if nans.any() and (~nans).sum() > 2:
            vals[nans] = np.interp(np.where(nans)[0], np.where(~nans)[0], vals[~nans])
    # Smooth
    for ax in range(3):
        wrist_3d[side][:, ax] = uniform_filter1d(wrist_3d[side][:, ax], size=9)

# ===== Step 5: Convert to MuJoCo frame and render =====
# R3D world frame is ARKit: Y=up, X=right, Z=toward user
# MuJoCo: Z=up, X=forward, Y=left
# Transform: mj_x = -ar_z + offset, mj_y = -ar_x, mj_z = ar_y

# Get camera mean position for centering
cam_positions = np.array([get_camera_transform(fi)[:3, 3] for fi in range(0, len(depth_files), max(1, len(depth_files)//50))])
cam_mean = cam_positions.mean(axis=0)
print(f"\nCamera mean position: {cam_mean}")

H1_HEAD_Z = 1.45
FORWARD_OFFSET = 0.4  # Larger offset for R3D data (hands are closer to camera)

mj_wrist = {}
for side in ['left', 'right']:
    rel = wrist_3d[side] - cam_mean
    mj = np.zeros_like(rel)
    mj[:, 0] = -rel[:, 2] * 1.5 + FORWARD_OFFSET  # forward = -arkit_z * scale + offset
    mj[:, 1] = rel[:, 0] * 1.5                    # left = +arkit_x * scale (amplify lateral motion)
    mj[:, 2] = -rel[:, 1] + 1.55             # up = -arkit_y + shoulder height (1.55 puts hands at chest height for table work)
    # Clamp X to reachable workspace
    mj[:, 0] = np.clip(mj[:, 0], 0.15, 0.7)
    mj_wrist[side] = mj

print("\nMuJoCo wrist ranges:")
for side in ['right', 'left']:
    print(f"  {side}: X=[{mj_wrist[side][:,0].min():.3f},{mj_wrist[side][:,0].max():.3f}] "
          f"Y=[{mj_wrist[side][:,1].min():.3f},{mj_wrist[side][:,1].max():.3f}] "
          f"Z=[{mj_wrist[side][:,2].min():.3f},{mj_wrist[side][:,2].max():.3f}]")

# ===== Step 6: Render =====
print("\nRendering...")
scene_xml = f"""
<mujoco>
  <include file="{str(ASSETS_DIR / 'robots' / 'h1hand_pos.xml')}"/>
  <visual><global offwidth="{WIDTH}" offheight="{HEIGHT}"/></visual>
  <asset>
    <texture type="2d" name="grid" builtin="checker" rgb1="0.9 0.9 0.9" rgb2="0.7 0.7 0.7" width="512" height="512"/>
    <material name="floor_mat" texture="grid" texrepeat="8 8"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <light pos="1 1 2" dir="-0.5 -0.5 -1" diffuse="0.4 0.4 0.4"/>
    <geom name="floor" type="plane" size="3 3 0.1" material="floor_mat"/>
  </worldbody>
</mujoco>
"""

m = mujoco.MjModel.from_xml_string(scene_xml)
d = mujoco.MjData(m)
renderer = mujoco.Renderer(m, width=WIDTH, height=HEIGHT)

joint_map = {}
for i in range(m.njnt):
    joint_map[m.joint(i).name] = m.jnt_qposadr[i]

ik = MuJoCoArmIK(m, d, joint_map)

# Load RGB frames for side-by-side
orig_frames = []
rgb_dir = 'r3d_output/stack2_rgb'
if os.path.exists(rgb_dir):
    for fn in sorted(os.listdir(rgb_dir))[:n_hamer]:
        img = cv2.imread(os.path.join(rgb_dir, fn))
        if img is not None:
            orig_frames.append(img)
print(f"Original frames: {len(orig_frames)}")

# Init standing
mujoco.mj_resetData(m, d)
d.qpos[0:3] = [0, 0, 0.98]
d.qpos[3:7] = [1, 0, 0, 0]
mujoco.mj_forward(m, d)

tmpdir = tempfile.mkdtemp(prefix="option3_")
n_render = min(n_hamer, len(orig_frames)) if orig_frames else n_hamer

print(f"Rendering {n_render} frames...")
for fi in range(n_render):
    # Set legs
    d.qpos[0:3] = [0, 0, 0.98]
    d.qpos[3:7] = [1, 0, 0, 0]
    for lj in ['left_hip_pitch', 'right_hip_pitch']:
        if lj in joint_map: d.qpos[joint_map[lj]] = -0.1
    for kj in ['left_knee', 'right_knee']:
        if kj in joint_map: d.qpos[joint_map[kj]] = 0.2
    
    # IK for arms
    for side in ['left', 'right']:
        if fi < len(mj_wrist[side]):
            ik.solve(mj_wrist[side][fi], side, n_iter=200, step_size=0.5)
            for ji, jn in enumerate(ARM_JOINTS[side]):
                if jn in joint_map:
                    d.qpos[joint_map[jn]] = d.qpos[ik.arm_qpos_idx[side][ji]]
    
    mujoco.mj_forward(m, d)
    
    # Render front view
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.3, 0, 1.05]
    cam.distance = 1.6; cam.azimuth = 180; cam.elevation = -8
    opt = mujoco.MjvOption()
    renderer.update_scene(d, camera=cam, scene_option=opt)
    renderer._scene.lights[0].pos[:] = [0,0,4]
    renderer._scene.lights[0].dir[:] = [0,0,-1]
    renderer._scene.lights[0].diffuse[:] = [0.9,0.9,0.9]
    renderer._scene.lights[0].ambient[:] = [0.6,0.6,0.6]
    if renderer._scene.nlight > 1:
        renderer._scene.lights[1].diffuse[:] = [0.7,0.7,0.7]
        renderer._scene.lights[1].ambient[:] = [0.4,0.4,0.4]
    robot_frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
    
    # Side-by-side
    if orig_frames and fi < len(orig_frames):
        orig = cv2.resize(orig_frames[fi], (WIDTH, HEIGHT))
        for img, label in [(orig, "Original R3D"), (robot_frame, "Pipeline (HaMeR+LiDAR)")]:
            cv2.rectangle(img, (5,5), (300,35), (0,0,0), -1)
            cv2.putText(img, label, (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        combined = np.hstack([orig, robot_frame])
    else:
        combined = robot_frame
    
    cv2.imwrite(os.path.join(tmpdir, f"frame_{fi:05d}.png"), combined)
    if fi % 60 == 0:
        print(f"  Frame {fi}/{n_render}")

renderer.close()
z.close()

output = "sim_renders/option3_stack2_lidar.mp4"
subprocess.run(["ffmpeg","-y","-framerate","30","-i",os.path.join(tmpdir,"frame_%05d.png"),
                "-c:v","libx264","-pix_fmt","yuv420p","-crf","23",output], capture_output=True)
shutil.rmtree(tmpdir)
print(f"Saved: {output}")
