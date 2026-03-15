"""
Compare GT replay vs HaMeR replay side by side.
Uses the SAME replay code that already works for GT, then adds HaMeR alongside.
"""
import json
import h5py
import numpy as np
import mujoco
import cv2
import os
import tempfile
import shutil
import subprocess
from pathlib import Path

# Import the working replay infrastructure
from replay_egodex_on_h1 import (
    load_egodex_episode, egodex_to_robot_frame, MuJoCoArmIK, 
    ARM_JOINTS, FINGER_CURL_JOINTS, fingertip_distance_to_curl
)

ASSETS_DIR = Path(__file__).parent / "humanoidbench_assets"
WIDTH, HEIGHT = 480, 360

# ---- Step 1: Load GT data using the WORKING path ----
print("Loading GT data (same path as working replay)...")
gt_data = load_egodex_episode('pick_place', 0)
gt_converted = egodex_to_robot_frame(gt_data)

# ---- Step 2: Build HaMeR data in the same format ----
print("Loading HaMeR data and converting...")

with open('egodex_sample/hamer_pick_place.json') as f:
    hamer = json.load(f)

gt_file = h5py.File('egodex_sample/test/basic_pick_place/0.hdf5', 'r')
tg = gt_file['transforms']
n_frames = tg['camera'].shape[0]
cam_transforms = tg['camera'][()]
focal_length = 736.6339
img_h = 1080

# Reconstruct HaMeR 3D in camera frame
hamer_cam = {'left': [], 'right': []}
hamer_ft_cam = {'left': [], 'right': []}

for fi in range(n_frames):
    frame_data = hamer.get(str(fi), {})
    hands = frame_data.get('hands', [])
    
    for side in ['left', 'right']:
        side_hands = [h for h in hands if h['side'] == side]
        if side_hands:
            h = side_hands[0]
            kps = np.array(h['keypoints_3d'])
            s, tx, ty = h['cam_t']
            
            z_depth = 2 * focal_length / (s * img_h)
            x_cam = tx / s + kps[0, 0] * z_depth
            y_cam = ty / s + kps[0, 1] * z_depth
            z_cam = z_depth + kps[0, 2] * z_depth
            
            hamer_cam[side].append([x_cam, y_cam, z_cam])
            
            ft_kps = kps[[4, 8, 12, 16, 20]]
            ft_3d = np.column_stack([
                tx/s + ft_kps[:, 0] * z_depth,
                ty/s + ft_kps[:, 1] * z_depth,
                z_depth + ft_kps[:, 2] * z_depth
            ])
            hamer_ft_cam[side].append(ft_3d)
        else:
            hamer_cam[side].append([np.nan]*3)
            hamer_ft_cam[side].append([[np.nan]*3]*5)

for side in ['left', 'right']:
    hamer_cam[side] = np.array(hamer_cam[side])
    hamer_ft_cam[side] = np.array(hamer_ft_cam[side])

# Get GT in camera frame for the affine fit
gt_cam = {'left': [], 'right': []}
for fi in range(n_frames):
    cam_inv = np.linalg.inv(cam_transforms[fi])
    for side in ['left', 'right']:
        pos = (cam_inv @ tg[f'{side}Hand'][fi])[:3, 3]
        gt_cam[side].append(pos)
for side in ['left', 'right']:
    gt_cam[side] = np.array(gt_cam[side])

gt_file.close()

# Fit SEPARATE affine transforms for each hand
from numpy.linalg import lstsq
A_b_per_hand = {}
for side in ['right', 'left']:
    valid = ~np.isnan(hamer_cam[side][:, 0])
    H = hamer_cam[side][valid]
    G = gt_cam[side][valid]
    H_aug = np.column_stack([H, np.ones(H.shape[0])])
    A_b_side, _, _, _ = lstsq(H_aug, G, rcond=None)
    A_b_per_hand[side] = A_b_side
    
    print(f"Affine fit ({side} hand):")
    for axis, name in enumerate(['X', 'Y', 'Z']):
        pred = (H_aug @ A_b_side)[:, axis]
        corr = np.corrcoef(pred, G[:, axis])[0, 1]
        rmse = np.sqrt(np.mean((pred - G[:, axis])**2))
        print(f"  {name}: corr={corr:.3f}, RMSE={rmse:.4f}m")

# Apply transform to get HaMeR in GT camera frame, then convert to world frame
# and then use the SAME egodex_to_robot_frame function
hamer_data = {
    'n_frames': n_frames,
    'description': gt_data['description'],
    'camera': gt_data['camera'],  # Same camera data
}

for side in ['left', 'right']:
    # Transform HaMeR cam → GT cam via affine
    valid_s = ~np.isnan(hamer_cam[side][:, 0])
    hamer_gt_cam = np.full_like(hamer_cam[side], np.nan)
    
    if valid_s.sum() > 0:
        H_s = np.column_stack([hamer_cam[side][valid_s], np.ones(valid_s.sum())])
        hamer_gt_cam[valid_s] = H_s @ A_b_per_hand[side]
    
    # Interpolate NaN frames
    for axis in range(3):
        vals = hamer_gt_cam[:, axis]
        nans = np.isnan(vals)
        if nans.any() and (~nans).sum() > 2:
            vals[nans] = np.interp(np.where(nans)[0], np.where(~nans)[0], vals[~nans])
    
    # Suppress HaMeR noise
    # The affine transform was fit on the RIGHT hand. For the left hand,
    # HaMeR's depth estimate is wildly unstable (scale varies 0.9-4.4).
    # Apply very heavy smoothing to suppress this.
    from scipy.ndimage import uniform_filter1d
    
    # Measure how much this hand actually moves (before smoothing)
    travel = np.sum(np.linalg.norm(np.diff(hamer_gt_cam, axis=0), axis=1))
    
    # Compare with GT travel to decide smoothing strength
    gt_travel_side = np.sum(np.linalg.norm(np.diff(gt_cam[side], axis=0), axis=1))
    motion_ratio = travel / (gt_travel_side + 1e-6)
    
    if motion_ratio > 5.0:
        # HaMeR is producing way more motion than GT — this hand is nearly stationary
        # Just use the median position (most robust to outliers)
        median_pos = np.median(hamer_gt_cam, axis=0)
        print(f"  {side}: HaMeR travel={travel:.2f}m vs GT={gt_travel_side:.2f}m (ratio={motion_ratio:.1f}x) → FREEZE to median {median_pos}")
        for fi_idx in range(n_frames):
            hamer_gt_cam[fi_idx] = median_pos
    else:
        # Light smoothing for the active hand
        smooth_size = 9
        print(f"  {side}: HaMeR travel={travel:.2f}m vs GT={gt_travel_side:.2f}m (ratio={motion_ratio:.1f}x) → light smoothing (window={smooth_size})")
        for axis in range(3):
            hamer_gt_cam[:, axis] = uniform_filter1d(hamer_gt_cam[:, axis], size=smooth_size)
    
    # Convert camera frame → ARKit world frame
    # cam_frame_pos = cam_inv @ world_pos, so world_pos = cam @ cam_frame_pos
    wrist_world = np.zeros((n_frames, 3))
    for fi in range(n_frames):
        cam_tf = cam_transforms[fi]
        wrist_world[fi] = cam_tf[:3, :3] @ hamer_gt_cam[fi] + cam_tf[:3, 3]
    
    hamer_data[f'{side}_wrist_pos'] = wrist_world
    hamer_data[f'{side}_wrist_rot'] = gt_data[f'{side}_wrist_rot']  # Use GT rotation for now
    
    # Fingertips: transform same way
    ft_world = np.zeros((n_frames, 5, 3))
    for fi in range(n_frames):
        valid_ft = ~np.isnan(hamer_ft_cam[side][fi, 0, 0])
        if valid_ft and valid_s[fi]:
            ft_cam = hamer_ft_cam[side][fi]  # 5x3
            ft_aug = np.column_stack([ft_cam, np.ones(5)])
            ft_gt_cam = ft_aug @ A_b_per_hand[side]  # 5x3
            cam_tf = cam_transforms[fi]
            for fti in range(5):
                ft_world[fi, fti] = cam_tf[:3, :3] @ ft_gt_cam[fti] + cam_tf[:3, 3]
        else:
            ft_world[fi] = gt_data[f'{side}_fingertips'][fi]  # Fallback to GT
    
    hamer_data[f'{side}_fingertips'] = ft_world

if 'mp4_path' in gt_data:
    hamer_data['mp4_path'] = gt_data['mp4_path']

# Convert HaMeR data through same robot-frame transform
hamer_converted = egodex_to_robot_frame(hamer_data)

print(f"\nGT MuJoCo right wrist range: "
      f"X=[{gt_converted['right_wrist_pos'][:,0].min():.3f},{gt_converted['right_wrist_pos'][:,0].max():.3f}] "
      f"Y=[{gt_converted['right_wrist_pos'][:,1].min():.3f},{gt_converted['right_wrist_pos'][:,1].max():.3f}]")
print(f"HaMeR MuJoCo right wrist range: "
      f"X=[{hamer_converted['right_wrist_pos'][:,0].min():.3f},{hamer_converted['right_wrist_pos'][:,0].max():.3f}] "
      f"Y=[{hamer_converted['right_wrist_pos'][:,1].min():.3f},{hamer_converted['right_wrist_pos'][:,1].max():.3f}]")

# ---- Step 3: Render both using same MuJoCo setup ----
print("\nRendering...")

scene_xml = f"""
<mujoco>
  <include file="{str(ASSETS_DIR / 'robots' / 'h1hand_pos.xml')}"/>
  <visual><global offwidth="{WIDTH}" offheight="{HEIGHT}"/></visual>
  <asset>
    <texture type="2d" name="grid" builtin="checker" rgb1="0.9 0.9 0.9" rgb2="0.7 0.7 0.7" width="512" height="512"/>
    <material name="floor_mat" texture="grid" texrepeat="8 8"/>
    <material name="table_mat" rgba="0.6 0.4 0.2 1"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <light pos="1 1 2" dir="-0.5 -0.5 -1" diffuse="0.4 0.4 0.4"/>
    <geom name="floor" type="plane" size="3 3 0.1" material="floor_mat"/>
  </worldbody>
</mujoco>
"""

# Two separate model instances so IK warm-starts are independent
m_gt = mujoco.MjModel.from_xml_string(scene_xml)
d_gt = mujoco.MjData(m_gt)
m_hm = mujoco.MjModel.from_xml_string(scene_xml)
d_hm = mujoco.MjData(m_hm)
renderer = mujoco.Renderer(m_gt, width=WIDTH, height=HEIGHT)

joint_map = {}
for i in range(m_gt.njnt):
    joint_map[m_gt.joint(i).name] = m_gt.jnt_qposadr[i]

ik_gt = MuJoCoArmIK(m_gt, d_gt, joint_map)
ik_hm = MuJoCoArmIK(m_hm, d_hm, joint_map)

orig_frames = []
cap = cv2.VideoCapture(gt_data.get('mp4_path', ''))
while True:
    ret, frame = cap.read()
    if not ret: break
    orig_frames.append(frame)
cap.release()

# Load DexMV finger data and smooth
dexmv_data = None
dexmv_path = 'retargeted/egodex_hamer_dexmv_fingers.npz'
if os.path.exists(dexmv_path):
    from scipy.ndimage import uniform_filter1d
    d_npz = np.load(dexmv_path, allow_pickle=True)
    dexmv_data = {'right': d_npz['right'].copy(), 'left': d_npz['left'].copy()}
    # Smooth finger DOFs to remove HaMeR frame-to-frame jitter
    smooth_window = 11  # ~0.37 seconds at 30fps
    for side in ['right', 'left']:
        for dof in range(6, 30):  # Only smooth finger DOFs (6-29), not arm DOFs (0-5)
            dexmv_data[side][:, dof] = uniform_filter1d(dexmv_data[side][:, dof], size=smooth_window)
    print(f"Loaded & smoothed DexMV fingers: right={dexmv_data['right'].shape}, left={dexmv_data['left'].shape}")
else:
    print("No DexMV data found, using distance heuristic for fingers")

tmpdir = tempfile.mkdtemp(prefix="compare_")

def render_robot(model, data, ik_solver, converted_data, fi, dexmv_data=None):
    """Set up robot and render. Keeps IK warm start from previous frame."""
    # Lock base and legs (don't reset arms — keep warm start)
    data.qpos[0:3] = [0, 0, 0.98]
    data.qpos[3:7] = [1, 0, 0, 0]
    for lj in ['left_hip_pitch', 'right_hip_pitch']:
        if lj in joint_map: data.qpos[joint_map[lj]] = -0.1
    for kj in ['left_knee', 'right_knee']:
        if kj in joint_map: data.qpos[joint_map[kj]] = 0.2
    
    # IK for arms (warm-started from previous frame)
    for side in ['left', 'right']:
        target = converted_data[f'{side}_wrist_pos'][fi]
        ik_solver.solve(target, side, n_iter=200, step_size=0.5)
        idxs = ik_solver.arm_qpos_idx[side]
        for ji, jname in enumerate(ARM_JOINTS[side]):
            if jname in joint_map:
                data.qpos[joint_map[jname]] = data.qpos[idxs[ji]]
    
    # Fingers — use DexMV if available, else distance heuristic
    if dexmv_data is not None:
        for side in ['left', 'right']:
            prefix = 'lh' if side == 'left' else 'rh'
            dexmv_joints = dexmv_data[side][fi]  # 30 DOF
            # Map Adroit DOF indices to Shadow Hand joints
            # Map Adroit curl DOFs only — skip spread/splay joints (FFJ0, MFJ0, RFJ0)
            # which cause unnatural finger divergence
            adroit_to_shadow = {
                6: 'WRJ1', 7: 'WRJ2',
                8: 'FFJ3', 9: 'FFJ2', 10: 'FFJ1',    # index curl (skip 11=FFJ0 spread)
                12: 'MFJ3', 13: 'MFJ2', 14: 'MFJ1',  # middle curl (skip 15=MFJ0 spread)
                16: 'RFJ3', 17: 'RFJ2', 18: 'RFJ1',  # ring curl (skip 19=RFJ0 spread)
                21: 'LFJ3', 22: 'LFJ2', 23: 'LFJ1',  # little curl (skip 20=LFJ5/24=LFJ0 spread)
                25: 'THJ5', 26: 'THJ4', 27: 'THJ3', 28: 'THJ2', 29: 'THJ1',  # thumb
            }
            for adroit_idx, shadow_suffix in adroit_to_shadow.items():
                jname = f'{prefix}_{shadow_suffix}'
                if jname in joint_map:
                    jnt_id = model.joint(jname).id
                    lo = model.jnt_range[jnt_id, 0]
                    hi = model.jnt_range[jnt_id, 1]
                    data.qpos[joint_map[jname]] = np.clip(dexmv_joints[adroit_idx], lo, hi)
    else:
        for side in ['left', 'right']:
            wrist_pos = converted_data[f'{side}_wrist_pos'][fi]
            fingertips = converted_data[f'{side}_fingertips'][fi]
            finger_names = ['index', 'middle', 'ring', 'little', 'thumb']
            for finger_idx, finger_name in enumerate(finger_names):
                curl = fingertip_distance_to_curl(wrist_pos, fingertips[finger_idx])
                for curl_joint in FINGER_CURL_JOINTS[side][finger_name]:
                    if curl_joint in joint_map:
                        jnt_id = model.joint(curl_joint).id
                        lo = model.jnt_range[jnt_id, 0]
                        hi = model.jnt_range[jnt_id, 1]
                        data.qpos[joint_map[curl_joint]] = lo + curl * (hi - lo) * 0.9
    
    mujoco.mj_forward(model, data)
    
    # Render front view
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.3, 0, 1.05]
    cam.distance = 1.6
    cam.azimuth = 180  # Front
    cam.elevation = -8
    opt = mujoco.MjvOption()
    renderer.update_scene(data, camera=cam, scene_option=opt)
    renderer._scene.lights[0].pos[:] = [0, 0, 4]
    renderer._scene.lights[0].dir[:] = [0, 0, -1]
    renderer._scene.lights[0].diffuse[:] = [0.9, 0.9, 0.9]
    renderer._scene.lights[0].ambient[:] = [0.6, 0.6, 0.6]
    if renderer._scene.nlight > 1:
        renderer._scene.lights[1].diffuse[:] = [0.7, 0.7, 0.7]
        renderer._scene.lights[1].ambient[:] = [0.4, 0.4, 0.4]
    pixels = renderer.render()
    return cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)

# Initialize both with standing pose
for dd in [d_gt, d_hm]:
    mujoco.mj_resetData(m_gt, dd)
    dd.qpos[0:3] = [0, 0, 0.98]
    dd.qpos[3:7] = [1, 0, 0, 0]
    mujoco.mj_forward(m_gt, dd)

for fi in range(n_frames):
    gt_frame = render_robot(m_gt, d_gt, ik_gt, gt_converted, fi, dexmv_data=None)  # GT uses distance heuristic
    hamer_frame = render_robot(m_hm, d_hm, ik_hm, hamer_converted, fi, dexmv_data=dexmv_data)  # HaMeR uses DexMV
    
    # Original video
    orig = cv2.resize(orig_frames[min(fi, len(orig_frames)-1)], (WIDTH, HEIGHT))
    
    # Labels
    for img, label in [(orig, "Original"), (gt_frame, "GT Replay"), (hamer_frame, "HaMeR Replay")]:
        cv2.rectangle(img, (5, 5), (220, 35), (0, 0, 0), -1)
        cv2.putText(img, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    combined = np.hstack([orig, gt_frame, hamer_frame])
    cv2.imwrite(os.path.join(tmpdir, f"frame_{fi:05d}.png"), combined)
    
    if fi % 30 == 0:
        print(f"  Frame {fi}/{n_frames}")

renderer.close()

output_path = "sim_renders/egodex_gt_vs_hamer_replay.mp4"
subprocess.run([
    "ffmpeg", "-y", "-framerate", "30",
    "-i", os.path.join(tmpdir, "frame_%05d.png"),
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
    output_path
], capture_output=True)
shutil.rmtree(tmpdir)
print(f"Saved: {output_path}")
