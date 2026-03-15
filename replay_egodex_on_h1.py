"""
Replay EgoDex ground-truth hand data on H1 + Shadow Hand in MuJoCo.

Takes the perfect, calibrated, world-frame 3D hand poses from EgoDex
and converts them to H1 joint targets via IK, then renders.

This tests whether our REPLAY code works given clean data.

Usage: python replay_egodex_on_h1.py [--episode pick_place|fold|stack]
"""
import h5py
import numpy as np
import mujoco
import cv2
import os
import argparse
from pathlib import Path

ASSETS_DIR = Path(__file__).parent / "humanoidbench_assets"
EGODEX_DIR = Path(__file__).parent / "egodex_sample"

# EgoDex fingertip joint names
FINGERTIP_JOINTS = {
    'left': ['leftThumbTip', 'leftIndexFingerTip', 'leftMiddleFingerTip',
             'leftRingFingerTip', 'leftLittleFingerTip'],
    'right': ['rightThumbTip', 'rightIndexFingerTip', 'rightMiddleFingerTip',
              'rightRingFingerTip', 'rightLittleFingerTip']
}

# Shadow Hand finger joint mapping (approximate curl from fingertip distance)
# FFJ = First Finger (Index), MFJ = Middle, RFJ = Ring, LFJ = Little, THJ = Thumb
FINGER_CURL_JOINTS = {
    'left': {
        'index': ['lh_FFJ3', 'lh_FFJ2', 'lh_FFJ1'],
        'middle': ['lh_MFJ3', 'lh_MFJ2', 'lh_MFJ1'],
        'ring': ['lh_RFJ3', 'lh_RFJ2', 'lh_RFJ1'],
        'little': ['lh_LFJ3', 'lh_LFJ2', 'lh_LFJ1'],
        'thumb': ['lh_THJ3', 'lh_THJ2', 'lh_THJ1'],
    },
    'right': {
        'index': ['rh_FFJ3', 'rh_FFJ2', 'rh_FFJ1'],
        'middle': ['rh_MFJ3', 'rh_MFJ2', 'rh_MFJ1'],
        'ring': ['rh_RFJ3', 'rh_RFJ2', 'rh_RFJ1'],
        'little': ['rh_LFJ3', 'rh_LFJ2', 'rh_LFJ1'],
        'thumb': ['rh_THJ3', 'rh_THJ2', 'rh_THJ1'],
    }
}

# H1 arm joints
ARM_JOINTS = {
    'left': ['left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw',
             'left_elbow', 'left_wrist_yaw'],
    'right': ['right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw',
              'right_elbow', 'right_wrist_yaw'],
}

def load_egodex_episode(episode_type, episode_idx=0):
    """Load EgoDex HDF5 data."""
    episode_map = {
        'pick_place': 'basic_pick_place',
        'fold': 'basic_fold',
        'stack': 'stack',
    }
    task = episode_map.get(episode_type, episode_type)
    hdf5_path = EGODEX_DIR / f"test/{task}/{episode_idx}.hdf5"
    mp4_path = EGODEX_DIR / f"test/{task}/{episode_idx}.mp4"
    
    f = h5py.File(str(hdf5_path), 'r')
    tg = f['transforms']
    n_frames = tg['camera'].shape[0]
    
    desc = f.attrs.get('llm_description', 'Unknown task')
    print(f"Episode: {task}/{episode_idx}")
    print(f"Task: {desc}")
    print(f"Frames: {n_frames}")
    
    # Extract wrist positions and fingertip positions
    data = {'n_frames': n_frames, 'description': str(desc)}
    
    for side in ['left', 'right']:
        hand_key = f"{side}Hand"
        hand_transforms = tg[hand_key][()]  # N x 4 x 4
        
        # Wrist position and rotation
        data[f'{side}_wrist_pos'] = hand_transforms[:, :3, 3]  # N x 3
        data[f'{side}_wrist_rot'] = hand_transforms[:, :3, :3]  # N x 3 x 3
        
        # Fingertip positions
        fingertips = []
        for ft_name in FINGERTIP_JOINTS[side]:
            ft_transforms = tg[ft_name][()]
            fingertips.append(ft_transforms[:, :3, 3])
        data[f'{side}_fingertips'] = np.stack(fingertips, axis=1)  # N x 5 x 3
    
    # Camera transforms (for reference frame)
    data['camera'] = tg['camera'][()]  # N x 4 x 4
    
    f.close()
    
    # Load video frames if available
    if mp4_path.exists():
        data['mp4_path'] = str(mp4_path)
    
    return data


def egodex_to_robot_frame(data):
    """
    Convert EgoDex world-frame coordinates to robot frame.
    
    EgoDex ARKit coordinates:
    - Y is up (gravity direction)
    - X is right (when facing forward)
    - Z is towards the user (into the screen)
    - Person sits at roughly: camera Y≈1.07, hands Y≈0.8
    
    MuJoCo H1:
    - Z is up
    - X is forward (robot faces +X)
    - Y is left
    - Robot stands at pelvis Z≈0.98, head Z≈1.55
    
    Strategy:
    1. Compute hand positions relative to camera (head)
    2. Apply axis rotation ARKit→MuJoCo
    3. Offset to H1 head position
    """
    cam_pos = data['camera'][:, :3, 3]  # N x 3, in ARKit world
    
    # H1 approximate body positions
    H1_HEAD_Z = 1.45  # head height in MuJoCo
    H1_TORSO_X = 0.0  # robot at origin
    H1_TORSO_Y = 0.0
    
    converted = {}
    converted['n_frames'] = data['n_frames']
    converted['description'] = data['description']
    
    def arkit_to_mujoco_vec(pts):
        """Convert (..., 3) from ARKit to MuJoCo coordinates.
        ARKit: X=right, Y=up, Z=towards-user
        MuJoCo: X=forward, Y=left, Z=up
        
        mj_x = -ar_z (forward = opposite of towards-user)
        mj_y = -ar_x (left = opposite of right)  
        mj_z = ar_y  (up = up)
        """
        result = np.zeros_like(pts)
        result[..., 0] = -pts[..., 2]
        result[..., 1] = -pts[..., 0]
        result[..., 2] = pts[..., 1]
        return result
    
    for side in ['left', 'right']:
        wrist_pos = data[f'{side}_wrist_pos'].copy()
        fingertips = data[f'{side}_fingertips'].copy()
        wrist_rot = data[f'{side}_wrist_rot'].copy()
        
        # Step 1: Make relative to camera (head) per frame
        wrist_rel = wrist_pos - cam_pos
        fingertips_rel = fingertips - cam_pos[:, None, :]
        
        # Step 2: Rotate axes
        wrist_mj = arkit_to_mujoco_vec(wrist_rel)
        fingertips_mj = arkit_to_mujoco_vec(fingertips_rel)
        
        # Step 3: Offset to H1 body position
        # Add forward offset so hands that are "below" the head (small/negative X) 
        # end up in front of the robot. The person's hands during tabletop tasks
        # are typically 0-30cm below the camera in ARKit Z, which maps to
        # 0 to -0.3 in MuJoCo X. Adding ~0.25m forward puts them in front.
        FORWARD_OFFSET = 0.25
        wrist_mj[..., 0] += H1_TORSO_X + FORWARD_OFFSET
        wrist_mj[..., 1] += H1_TORSO_Y
        wrist_mj[..., 2] += H1_HEAD_Z
        
        fingertips_mj[..., 0] += H1_TORSO_X + FORWARD_OFFSET
        fingertips_mj[..., 1] += H1_TORSO_Y
        fingertips_mj[..., 2] += H1_HEAD_Z
        
        converted[f'{side}_wrist_pos'] = wrist_mj
        converted[f'{side}_fingertips'] = fingertips_mj
        
        # Rotate the rotation matrix too
        R_transform = np.array([
            [0, 0, -1],
            [-1, 0, 0],
            [0, 1, 0]
        ], dtype=np.float32)
        converted[f'{side}_wrist_rot'] = np.array([R_transform @ R for R in wrist_rot])
    
    if 'mp4_path' in data:
        converted['mp4_path'] = data['mp4_path']
    
    return converted


def fingertip_distance_to_curl(wrist_pos, fingertip_pos, max_dist=0.17, min_dist=0.08):
    """
    Convert fingertip-to-wrist distance to a curl value (0=extended, 1=fully curled).
    EgoDex fingertip distances: ~0.13-0.17m when open, ~0.08-0.11m when grasping.
    """
    dist = np.linalg.norm(fingertip_pos - wrist_pos)
    curl = 1.0 - np.clip((dist - min_dist) / (max_dist - min_dist), 0, 1)
    return curl


class MuJoCoArmIK:
    """IK solver using MuJoCo's Jacobian for H1 arms."""
    
    def __init__(self, model, data, joint_map):
        self.m = model
        self.d = data
        self.joint_map = joint_map
        
        # Arm joint names and indices
        self.arm_joints = {
            'left': ['left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw',
                     'left_elbow', 'left_wrist_yaw'],
            'right': ['right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw',
                      'right_elbow', 'right_wrist_yaw'],
        }
        # End effector body names  
        self.ee_body = {'left': 'lh_palm', 'right': 'rh_palm'}
        
        # Cache joint qpos indices and ranges
        self.arm_qpos_idx = {}
        self.arm_ranges = {}
        for side in ['left', 'right']:
            idxs = [joint_map[j] for j in self.arm_joints[side]]
            self.arm_qpos_idx[side] = idxs
            ranges = []
            for j in self.arm_joints[side]:
                jid = model.joint(j).id
                ranges.append(model.jnt_range[jid].copy())
            self.arm_ranges[side] = np.array(ranges)  # 5x2
    
    def solve(self, target_pos, side, n_iter=100, step_size=0.2):
        """Solve IK for one arm. Returns joint angles for the 5 arm joints."""
        ee_bid = self.m.body(self.ee_body[side]).id
        idxs = self.arm_qpos_idx[side]
        ranges = self.arm_ranges[side]
        
        for iteration in range(n_iter):
            mujoco.mj_forward(self.m, self.d)
            
            # Current EE position
            ee_pos = self.d.xpos[ee_bid].copy()
            error = target_pos - ee_pos
            
            if np.linalg.norm(error) < 0.005:  # 5mm tolerance
                break
            
            # Compute Jacobian for EE body
            jacp = np.zeros((3, self.m.nv))
            mujoco.mj_jacBody(self.m, self.d, jacp, None, ee_bid)
            
            # Extract columns for arm joints only
            arm_dof_idxs = [self.joint_map[j] - 1 for j in self.arm_joints[side]]
            # Actually qpos != dof for free joint. Free joint has qpos=7, dof=6
            # For hinge joints after free joint: dof_idx = qpos_idx - 1
            arm_dof_idxs = []
            for j in self.arm_joints[side]:
                jid = self.m.joint(j).id
                arm_dof_idxs.append(self.m.jnt_dofadr[jid])
            
            J = jacp[:, arm_dof_idxs]  # 3x5
            
            # Damped pseudoinverse
            damping = 0.01
            JtJ = J.T @ J + damping * np.eye(J.shape[1])
            dq = np.linalg.solve(JtJ, J.T @ error) * step_size
            
            # Apply and clamp
            for i, idx in enumerate(idxs):
                self.d.qpos[idx] += dq[i]
                self.d.qpos[idx] = np.clip(self.d.qpos[idx], ranges[i, 0], ranges[i, 1])
        
        return np.array([self.d.qpos[idx] for idx in idxs])


def render_replay(converted_data, output_path, width=640, height=480):
    """Render the replay as a video."""
    
    # Create a scene XML that includes H1 + floor + table + good lighting
    scene_xml = f"""
    <mujoco>
      <include file="{str(ASSETS_DIR / 'robots' / 'h1hand_pos.xml')}"/>
      <visual>
        <global offwidth="{width}" offheight="{height}"/>
        <quality shadowsize="2048"/>
        <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0.2 0.2 0.2"/>
      </visual>
      <asset>
        <texture type="2d" name="grid" builtin="checker" rgb1="0.9 0.9 0.9" rgb2="0.7 0.7 0.7" width="512" height="512"/>
        <material name="floor_mat" texture="grid" texrepeat="8 8" reflectance="0.1"/>
        <material name="table_mat" rgba="0.6 0.4 0.2 1"/>
      </asset>
      <worldbody>
        <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8" specular="0.3 0.3 0.3"/>
        <light pos="1 1 2" dir="-0.5 -0.5 -1" diffuse="0.4 0.4 0.4"/>
        <geom name="floor" type="plane" size="3 3 0.1" material="floor_mat" pos="0 0 0"/>
        <body name="table" pos="0.5 0 0.4">
          <geom type="box" size="0.4 0.6 0.02" material="table_mat"/>
          <geom type="cylinder" size="0.03 0.2" pos="0.3 0.5 -0.2" material="table_mat"/>
          <geom type="cylinder" size="0.03 0.2" pos="-0.3 0.5 -0.2" material="table_mat"/>
          <geom type="cylinder" size="0.03 0.2" pos="0.3 -0.5 -0.2" material="table_mat"/>
          <geom type="cylinder" size="0.03 0.2" pos="-0.3 -0.5 -0.2" material="table_mat"/>
        </body>
      </worldbody>
    </mujoco>
    """
    
    # Try loading the combined scene
    try:
        m = mujoco.MjModel.from_xml_string(scene_xml)
        print(f"Loaded combined scene: nq={m.nq}, njnt={m.njnt}")
    except Exception as e:
        print(f"Scene XML failed ({e}), falling back to robot-only model")
        model_path = str(ASSETS_DIR / "robots" / "h1hand_pos.xml")
        m = mujoco.MjModel.from_xml_path(model_path)
        print(f"Loaded robot-only: nq={m.nq}, njnt={m.njnt}")
    d = mujoco.MjData(m)
    
    # Build joint name → qpos index mapping
    joint_map = {}
    for i in range(m.njnt):
        name = m.joint(i).name
        joint_map[name] = m.jnt_qposadr[i]
    
    # Create IK solver
    ik_solver = MuJoCoArmIK(m, d, joint_map)
    
    # Setup renderer
    renderer = mujoco.Renderer(m, width=width, height=height)
    
    n_frames = converted_data['n_frames']
    
    # Load original video for side-by-side if available
    orig_frames = []
    if 'mp4_path' in converted_data:
        cap = cv2.VideoCapture(converted_data['mp4_path'])
        while True:
            ret, frame = cap.read()
            if not ret: break
            orig_frames.append(frame)
        cap.release()
        print(f"Loaded {len(orig_frames)} original frames")
    
    # Write frames to temp dir then stitch with ffmpeg for H264 compatibility
    side_by_side = len(orig_frames) >= n_frames
    out_width = width * 2 if side_by_side else width  # Will be overridden for 3-panel
    
    import tempfile, shutil, subprocess
    tmpdir = tempfile.mkdtemp(prefix="egodex_render_")
    print(f"Rendering to temp dir: {tmpdir}")
    
    print(f"Rendering {n_frames} frames...")
    
    # Initialize standing pose once
    mujoco.mj_resetData(m, d)
    d.qpos[0:3] = [0, 0, 0.98]
    d.qpos[3:7] = [1, 0, 0, 0]
    for leg_joint in ['left_hip_pitch', 'right_hip_pitch']:
        if leg_joint in joint_map:
            d.qpos[joint_map[leg_joint]] = -0.1
    for knee_joint in ['left_knee', 'right_knee']:
        if knee_joint in joint_map:
            d.qpos[joint_map[knee_joint]] = 0.2
    mujoco.mj_forward(m, d)
    
    # Store previous arm solutions for warm-starting IK
    prev_arm = {'left': None, 'right': None}
    
    for fi in range(n_frames):
        # DON'T reset — keep previous frame's arm positions as IK warm start
        # Just re-lock the base and legs
        d.qpos[0:3] = [0, 0, 0.98]
        d.qpos[3:7] = [1, 0, 0, 0]
        for leg_joint in ['left_hip_pitch', 'right_hip_pitch']:
            if leg_joint in joint_map:
                d.qpos[joint_map[leg_joint]] = -0.1
        for knee_joint in ['left_knee', 'right_knee']:
            if knee_joint in joint_map:
                d.qpos[joint_map[knee_joint]] = 0.2
        
        # Set arm positions from Jacobian IK (warm-started from previous frame)
        for side in ['left', 'right']:
            wrist_target = converted_data[f'{side}_wrist_pos'][fi]
            arm_angles = ik_solver.solve(wrist_target, side, n_iter=200, step_size=0.5)
            
            for ji, jname in enumerate(ARM_JOINTS[side]):
                if jname in joint_map:
                    d.qpos[joint_map[jname]] = arm_angles[ji]
        
        # Set wrist orientation from EgoDex rotation data
        for side in ['left', 'right']:
            wrist_rot = converted_data[f'{side}_wrist_rot'][fi]  # 3x3 rotation matrix in MuJoCo frame
            
            # Use wrist rotation to set wrist_yaw and WRJ1/WRJ2
            # Extract palm facing direction (rotation matrix Z column = palm normal)
            palm_dir = wrist_rot[:, 2]  # Z column of rotation = direction palm faces
            
            # WRJ1 controls wrist flexion (palm up/down) 
            # WRJ2 controls wrist deviation (palm left/right)
            prefix = 'lh' if side == 'left' else 'rh'
            
            # Compute wrist angles from palm direction
            # palm_dir[2] < 0 means palm faces down (desired for tabletop manipulation)
            wrj1_angle = np.arctan2(-palm_dir[2], palm_dir[0]) * 0.5  # flexion
            wrj2_angle = np.arctan2(palm_dir[1], palm_dir[0]) * 0.3  # deviation
            
            wrj1_name = f'{prefix}_WRJ1'
            wrj2_name = f'{prefix}_WRJ2'
            if wrj1_name in joint_map:
                jid = m.joint(wrj1_name).id
                lo, hi = m.jnt_range[jid]
                d.qpos[joint_map[wrj1_name]] = np.clip(wrj1_angle, lo, hi)
            if wrj2_name in joint_map:
                jid = m.joint(wrj2_name).id
                lo, hi = m.jnt_range[jid]
                d.qpos[joint_map[wrj2_name]] = np.clip(wrj2_angle, lo, hi)
        
        # Set finger curls from fingertip distances (for BOTH hands)
        for side in ['left', 'right']:
            wrist_pos = converted_data[f'{side}_wrist_pos'][fi]
            fingertips = converted_data[f'{side}_fingertips'][fi]  # 5 x 3
            
            finger_names = ['index', 'middle', 'ring', 'little', 'thumb']
            
            for finger_idx, finger_name in enumerate(finger_names):
                curl = fingertip_distance_to_curl(wrist_pos, fingertips[finger_idx])
                
                for curl_joint in FINGER_CURL_JOINTS[side][finger_name]:
                    if curl_joint in joint_map:
                        jnt_id = m.joint(curl_joint).id
                        lo = m.jnt_range[jnt_id, 0]
                        hi = m.jnt_range[jnt_id, 1]
                        d.qpos[joint_map[curl_joint]] = lo + curl * (hi - lo) * 0.9
        
        # Debug: print EE position for key frames
        if fi < 3 or fi % 30 == 0:
            mujoco.mj_forward(m, d)
            for side, body_name in [('right', 'rh_palm')]:
                try:
                    bid = m.body(body_name).id
                    ee = d.xpos[bid]
                    tgt = converted_data[f'{side}_wrist_pos'][fi]
                    err = np.linalg.norm(tgt - ee)
                    print(f"  f{fi} {side}: target={tgt}, ee={ee}, err={err:.4f}")
                except:
                    pass
        
        # Forward kinematics
        mujoco.mj_forward(m, d)
        
        # Render helper
        def render_view(azimuth, elevation=-8, distance=1.6, lookat=[0.3, 0, 1.05]):
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = lookat
            cam.distance = distance
            cam.azimuth = azimuth
            cam.elevation = elevation
            opt = mujoco.MjvOption()
            renderer.update_scene(d, camera=cam, scene_option=opt)
            renderer._scene.lights[0].pos[:] = [0, 0, 4]
            renderer._scene.lights[0].dir[:] = [0, 0, -1]
            renderer._scene.lights[0].diffuse[:] = [0.9, 0.9, 0.9]
            renderer._scene.lights[0].ambient[:] = [0.6, 0.6, 0.6]
            renderer._scene.lights[0].specular[:] = [0.3, 0.3, 0.3]
            if renderer._scene.nlight > 1:
                renderer._scene.lights[1].pos[:] = [-1, -1, 3]
                renderer._scene.lights[1].dir[:] = [0.3, 0.3, -1]
                renderer._scene.lights[1].diffuse[:] = [0.7, 0.7, 0.7]
                renderer._scene.lights[1].ambient[:] = [0.4, 0.4, 0.4]
            pixels = renderer.render()
            return cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        
        # Render both front and rear views
        # azimuth=0 means camera is at +X looking toward origin → sees robot's BACK
        # azimuth=180 means camera is at -X looking toward origin → sees robot's FRONT
        frame_front = render_view(azimuth=180)  # Front — face the robot
        frame_rear = render_view(azimuth=0)     # Rear — same direction as egocentric
        
        if side_by_side and fi < len(orig_frames):
            # 3-panel: Original | Robot Front | Robot Rear
            panel_w = width // 3 * 2  # Make original wider
            orig = orig_frames[fi]
            orig_resized = cv2.resize(orig, (width, height))
            
            # Resize robot panels to fit in remaining space
            robot_w = width // 2
            front_resized = cv2.resize(frame_front, (robot_w, height))
            rear_resized = cv2.resize(frame_rear, (robot_w, height))
            
            # Add labels
            labels = [
                (orig_resized, "Original (EgoDex)"),
                (front_resized, "Robot Front"),
                (rear_resized, "Robot Rear"),
            ]
            for img, label in labels:
                cv2.rectangle(img, (5, 5), (min(250, img.shape[1]-5), 40), (0, 0, 0), -1)
                cv2.putText(img, label, (10, 32),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            combined = np.hstack([orig_resized, front_resized, rear_resized])
            cv2.imwrite(os.path.join(tmpdir, f"frame_{fi:05d}.png"), combined)
        else:
            cv2.imwrite(os.path.join(tmpdir, f"frame_{fi:05d}.png"), frame_front)
        
        if fi % 30 == 0:
            print(f"  Frame {fi}/{n_frames}")
    
    renderer.close()
    
    # Stitch with ffmpeg for H264 codec
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-framerate", "30",
        "-i", os.path.join(tmpdir, "frame_%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "23", "-preset", "fast",
        str(output_path)
    ]
    print(f"Stitching with ffmpeg...")
    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg failed: {result.stderr[:500]}")
        # Fallback to mp4v
        print("Falling back to mp4v codec...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, 30.0, (out_width, height))
        for fi in range(n_frames):
            img = cv2.imread(os.path.join(tmpdir, f"frame_{fi:05d}.png"))
            if img is not None:
                out.write(img)
        out.release()
    
    shutil.rmtree(tmpdir)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', default='pick_place', choices=['pick_place', 'fold', 'stack'])
    parser.add_argument('--idx', type=int, default=0)
    args = parser.parse_args()
    
    # Load EgoDex data
    data = load_egodex_episode(args.episode, args.idx)
    
    # Print trajectory stats
    for side in ['left', 'right']:
        pos = data[f'{side}_wrist_pos']
        travel = np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1))
        print(f"{side} wrist travel: {travel:.3f}m")
    
    # Convert to robot frame
    converted = egodex_to_robot_frame(data)
    
    # Print converted stats
    print("\nConverted wrist ranges (MuJoCo frame):")
    for side in ['left', 'right']:
        pos = converted[f'{side}_wrist_pos']
        print(f"  {side}: X=[{pos[:,0].min():.3f}, {pos[:,0].max():.3f}] "
              f"Y=[{pos[:,1].min():.3f}, {pos[:,1].max():.3f}] "
              f"Z=[{pos[:,2].min():.3f}, {pos[:,2].max():.3f}]")
    
    # Render
    os.makedirs('sim_renders', exist_ok=True)
    output_path = f"sim_renders/egodex_{args.episode}_replay.mp4"
    render_replay(converted, output_path)


if __name__ == '__main__':
    main()
