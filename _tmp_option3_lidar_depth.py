"""
Option 3: HaMeR 2D keypoints + LiDAR depth from R3D.

Strategy:
1. From HaMeR: get 2D wrist pixel location in the image
2. From R3D depth map: get the depth at that pixel
3. Combine with camera intrinsics to get 3D world position
4. Use R3D camera poses to transform to world frame
"""
import json, numpy as np, os, struct, zipfile
from pathlib import Path

# Load HaMeR results
with open('modal_results/stack2_hamer_bimanual.json') as f:
    hamer_data = json.load(f)
results = hamer_data['results']
print(f"HaMeR: {len(results)} frames, bimanual={hamer_data['bimanual_frames']}")

# Load R3D metadata to get camera intrinsics and poses
r3d_path = 'raw_captures/stack2/stack2.r3d'

print(f"R3D file: {r3d_path}")

# Parse R3D metadata
with zipfile.ZipFile(r3d_path, 'r') as z:
    names = z.namelist()
    print(f"R3D entries: {len(names)}")
    
    # Get metadata
    if 'metadata' in names:
        meta_bytes = z.read('metadata')
        meta = json.loads(meta_bytes)
        print(f"Metadata keys: {list(meta.keys())}")
        
        # Camera intrinsics
        K = np.array(meta.get('K', meta.get('intrinsicMatrix', [])))
        if K.size == 9:
            K = K.reshape(3, 3)
        elif K.size > 0:
            K = np.array(K).reshape(3, 3)
        print(f"Camera K:\n{K}")
        
        # Image dimensions
        w = meta.get('w', meta.get('width', 960))
        h = meta.get('h', meta.get('height', 720))
        print(f"Image size: {w}x{h}")
        
        # Poses
        poses = meta.get('poses', [])
        n_poses = len(poses) // 7 if isinstance(poses, list) else 0
        print(f"Poses: {n_poses} frames (7 values each: qx,qy,qz,qw,tx,ty,tz)")
        
        # Depth info
        dw = meta.get('dw', meta.get('depthWidth', 256))
        dh = meta.get('dh', meta.get('depthHeight', 192))
        print(f"Depth size: {dw}x{dh}")
    
    # Check for depth files
    depth_files = [n for n in names if 'depth' in n.lower() or n.endswith('.depth')]
    conf_files = [n for n in names if 'conf' in n.lower()]
    rgb_files = [n for n in names if n.endswith('.jpg') or n.endswith('.png')]
    print(f"Depth files: {len(depth_files)}")
    print(f"Confidence files: {len(conf_files)}")
    print(f"RGB files: {len(rgb_files)}")
    
    # Try to load one depth frame
    if depth_files:
        depth_bytes = z.read(depth_files[0])
        print(f"Depth file 0: {depth_files[0]}, {len(depth_bytes)} bytes")
        # R3D depth is stored as float32 array
        depth_arr = np.frombuffer(depth_bytes, dtype=np.float32)
        print(f"Depth array: {depth_arr.shape}, range=[{depth_arr.min():.3f}, {depth_arr.max():.3f}]")
        if dw * dh == len(depth_arr):
            depth_map = depth_arr.reshape(dh, dw)
            print(f"Depth map: {depth_map.shape}")
            print(f"Depth center: {depth_map[dh//2, dw//2]:.3f}m")

# Now: for each HaMeR frame, get the 2D wrist position
# HaMeR's cam_crop_to_full gives us the full-image camera translation
# But we want the 2D pixel position of the wrist in the original image
# This comes from the bbox + cam_t

HAMER_FOCAL = 5000.0

print("\n=== HaMeR wrist 2D positions ===")
for fi in [0, 50, 100, 200, 300, 400]:
    if fi >= len(results): break
    frame = results[fi]
    for hand in frame['hands']:
        side = 'right' if hand.get('hand', 0) == 1 or hand.get('bbox_center_x', 0) < w/2 else 'left'
        kps = np.array(hand['keypoints_3d'])  # 21x3 in crop-relative
        s, tx, ty = hand['cam_t']
        bbox = hand['bbox']  # [x1, y1, x2, y2]
        x1, y1, x2, y2 = bbox
        cx, cy = (x1+x2)/2, (y1+y2)/2
        box_size = max(x2-x1, y2-y1)
        
        # Wrist 2D position in full image
        # In crop space: kp_2d = s * kp_3d[:2] + [tx, ty]
        # Crop covers [cx-box_size/2, cy-box_size/2] to [cx+box_size/2, cy+box_size/2]
        # Full image pixel = crop_pixel * box_size/crop_res + (cx - box_size/2)
        # HaMeR crop_res = 256 typically
        crop_res = 256
        wrist_crop_x = (s * kps[0, 0] + tx + 1) / 2 * crop_res  # normalized [-1,1] → [0, crop_res]
        wrist_crop_y = (s * kps[0, 1] + ty + 1) / 2 * crop_res
        wrist_img_x = wrist_crop_x / crop_res * box_size + (cx - box_size/2)
        wrist_img_y = wrist_crop_y / crop_res * box_size + (cy - box_size/2)
        
        # Depth lookup pixel (depth is at lower resolution)
        depth_x = int(wrist_img_x * dw / w)
        depth_y = int(wrist_img_y * dh / h)
        
        print(f"  f{fi} {side}: bbox=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}] wrist_2d=({wrist_img_x:.1f},{wrist_img_y:.1f}) depth_px=({depth_x},{depth_y})")
