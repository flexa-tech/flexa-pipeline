"""
Convert HaMeR MANO output to DexMV format (16 joint SE3 transforms).
"""
import json
import numpy as np
import os
from scipy.spatial.transform import Rotation

with open('egodex_sample/hamer_pick_place.json') as f:
    hamer = json.load(f)

n_frames = len(hamer)
print(f"Frames: {n_frames}")

def rotmat_to_se3(R, t=None):
    """Create 4x4 SE3 from 3x3 rotation and 3-vector translation."""
    T = np.eye(4)
    T[:3, :3] = R
    if t is not None:
        T[:3, 3] = t
    return T

for side in ['right', 'left']:
    out_dir = f'dexmv_input/egodex_hamer/{side}'
    os.makedirs(out_dir, exist_ok=True)
    
    count = 0
    for fi in range(n_frames):
        frame_data = hamer.get(str(fi), {})
        hands = frame_data.get('hands', [])
        side_hands = [h for h in hands if h['side'] == side]
        
        if not side_hands:
            joints = np.zeros((21, 3))
            # 16 identity SE3 transforms
            global_transforms = np.tile(np.eye(4), (16, 1, 1))
        else:
            h = side_hands[0]
            joints = np.array(h['keypoints_3d'])  # 21x3
            
            # Build 16 joint SE3 transforms from MANO params
            # MANO has 1 global orient + 15 hand pose = 16 joints
            global_orient = np.array(h['mano_global_orient'][0])  # 3x3 rotation matrix
            hand_pose = np.array(h['mano_hand_pose'])  # 15x3x3 rotation matrices
            
            global_transforms = np.zeros((16, 4, 4))
            
            # Joint 0: global orient with wrist position
            global_transforms[0] = rotmat_to_se3(global_orient, joints[0])
            
            # Joints 1-15: hand pose joints with corresponding keypoint positions
            # MANO joint order: wrist, index(3), middle(3), pinky(3), ring(3), thumb(3)
            # Keypoint positions from keypoints_3d correspond to these
            mano_to_kp = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]
            for ji in range(15):
                kp_idx = mano_to_kp[ji + 1] if ji + 1 < len(mano_to_kp) else 0
                global_transforms[ji + 1] = rotmat_to_se3(
                    hand_pose[ji], 
                    joints[min(kp_idx, 20)]
                )
        
        np.save(os.path.join(out_dir, f'joints_{fi}.npy'), joints.astype(np.float32))
        np.save(os.path.join(out_dir, f'results_global_{fi}.npy'), global_transforms.astype(np.float32))
        count += 1
    
    print(f"{side}: saved {count} frames to {out_dir}")
    # Verify shapes
    test_j = np.load(os.path.join(out_dir, 'joints_0.npy'))
    test_g = np.load(os.path.join(out_dir, 'results_global_0.npy'))
    print(f"  joints shape: {test_j.shape}, global shape: {test_g.shape}")

print("Ready for DexMV retargeting")
