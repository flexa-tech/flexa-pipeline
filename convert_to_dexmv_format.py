"""Convert HaMeR MANO output to DexMV input format.

DexMV expects per-frame .npy files:
  - joints_N.npy: (21, 3) MANO joint positions in WORLD frame
  - results_global_N.npy: (16, 4, 4) per-bone 4x4 transforms in WORLD frame

We:
1. Get HaMeR keypoints_3d (camera-relative) + MANO pose params
2. Transform to world frame using R3D camera poses
3. Compute per-bone transforms from MANO forward kinematics
4. Save in DexMV's format

Usage: python convert_to_dexmv_format.py stack2
"""
import json, sys, zipfile, numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation

PIPELINE = Path(__file__).parent
RAW = PIPELINE / "raw_captures"
MODAL = PIPELINE / "modal_results"
DEXMV_DATA = PIPELINE / "dexmv_input"


def mano_fk(global_orient_mat, hand_pose_mats, keypoints):
    """Compute approximate per-bone world transforms from MANO params.
    
    Returns (16, 4, 4) array — one 4x4 transform per bone.
    MANO has 16 bones: 1 wrist + 15 finger joints (3 per finger × 5 fingers).
    """
    # bone_to_kp mapping: bone i corresponds to keypoint
    # MANO kinematic tree: wrist→thumb(1,2,3,4)→index(5,6,7,8)→middle(9,10,11,12)→ring(13,14,15,16)→pinky(17,18,19,20)
    # 16 bones = wrist + 15 finger bones
    # Each bone's transform = rotation + translation (from parent joint position)
    
    bone_transforms = np.zeros((16, 4, 4))
    
    # Bone 0: wrist (root)
    T0 = np.eye(4)
    T0[:3, :3] = global_orient_mat
    T0[:3, 3] = keypoints[0]  # wrist position
    bone_transforms[0] = T0
    
    # Finger bones 1-15 (from hand_pose_mats [15, 3, 3])
    # Map finger bone index to MANO keypoint pairs
    bone_to_kp = [
        (0, 1), (1, 2), (2, 3),   # thumb
        (0, 5), (5, 6), (6, 7),   # index
        (0, 9), (9, 10), (10, 11), # middle
        (0, 13), (13, 14), (14, 15), # ring
        (0, 17), (17, 18), (18, 19), # pinky
    ]
    
    for i, (parent_kp, child_kp) in enumerate(bone_to_kp):
        T = np.eye(4)
        T[:3, :3] = hand_pose_mats[i] if i < len(hand_pose_mats) else np.eye(3)
        T[:3, 3] = keypoints[child_kp]
        bone_transforms[i + 1] = T
    
    return bone_transforms


def load_camera_poses(r3d_path):
    with zipfile.ZipFile(r3d_path) as z:
        meta = json.load(z.open("metadata"))
    poses = meta["poses"]  # list of 7-element [qx,qy,qz,qw,tx,ty,tz]
    return poses, meta


def pose_to_transform(p):
    """7-element pose → 4x4 transform matrix."""
    quat = p[:4]  # qx, qy, qz, qw
    trans = p[4:]
    R = Rotation.from_quat(quat).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = trans
    return T


def run(session="stack2"):
    r3d_path = RAW / session / f"{session}.r3d"
    hamer_path = MODAL / f"{session}_hamer_bimanual.json"
    
    if not hamer_path.exists():
        print(f"HaMeR data not found: {hamer_path}")
        return
    
    print(f"Loading HaMeR data ({hamer_path.stat().st_size//1024}KB)...")
    data = json.loads(hamer_path.read_text())
    
    poses, meta = load_camera_poses(r3d_path)
    print(f"Camera poses: {len(poses)}, Frame size: {meta['w']}x{meta['h']}")
    
    # Process both hands separately
    for hand_side in ["right", "left"]:
        out_dir = DEXMV_DATA / session / hand_side
        out_dir.mkdir(parents=True, exist_ok=True)
        
        valid_count = 0
        for r in data["results"]:
            frame_idx = r["frame_idx"]
            
            # Get hand data
            hand_data = next((h for h in r.get("hands", []) if h.get("hand") == hand_side), None)
            if hand_data is None:
                continue
            
            kp3d = np.array(hand_data["keypoints_3d"], dtype=np.float32)  # (21, 3) camera-relative
            hand_pose = hand_data.get("mano_hand_pose")
            global_orient = hand_data.get("mano_global_orient")
            
            if hand_pose is None or global_orient is None:
                continue
            
            hand_pose_mats = np.array(hand_pose, dtype=np.float32)  # (15, 3, 3)
            global_orient_mat = np.array(global_orient, dtype=np.float32).reshape(3, 3)  # (3, 3)
            
            # Get camera pose for this frame
            pose_idx = min(frame_idx, len(poses) - 1)
            cam_T = pose_to_transform(poses[pose_idx])  # camera-to-world
            cam_R = cam_T[:3, :3]
            cam_t = cam_T[:3, 3]
            
            # Transform keypoints to world frame
            # kp3d is in camera frame: world = R @ cam + t
            kp3d_world = (cam_R @ kp3d.T).T + cam_t  # (21, 3)
            
            # Transform global_orient to world frame
            global_orient_world = cam_R @ global_orient_mat  # (3, 3)
            
            # Transform hand_pose_mats (these are local rotations, keep relative but apply world rotation to first)
            hand_pose_world = np.zeros_like(hand_pose_mats)
            for i in range(15):
                hand_pose_world[i] = cam_R @ hand_pose_mats[i]
            
            # Compute bone transforms
            bone_transforms = mano_fk(global_orient_world, hand_pose_world, kp3d_world)
            
            # Save as DexMV format
            frame_num = frame_idx + 1  # DexMV uses 1-indexed
            np.save(str(out_dir / f"joints_{frame_num}.npy"), kp3d_world.astype(np.float32))
            np.save(str(out_dir / f"results_global_{frame_num}.npy"), bone_transforms.astype(np.float32))
            
            valid_count += 1
        
        print(f"  {hand_side}: {valid_count} frames saved to {out_dir}")
    
    print(f"\nDone! DexMV input data at: {DEXMV_DATA / session}")
    print(f"To run DexMV retargeting:")
    print(f"  cd _dexmv/examples")
    print(f"  python retarget_human_hand.py --hand_dir={DEXMV_DATA / session / 'right'} --output_file=stack2_right.pkl")


if __name__ == "__main__":
    session = sys.argv[1] if len(sys.argv) > 1 else "stack2"
    run(session)
