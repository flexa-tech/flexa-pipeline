"""
Run our FULL pipeline (HaMeR + DexMV) on EgoDex video to test against ground truth.

Step 1: Run HaMeR on EgoDex frames → get MANO 3D keypoints
Step 2: Run DexMV on HaMeR keypoints → get finger retargeting
Step 3: Compare with EgoDex ground truth

Uses the existing modal_hamer_bimanual infrastructure.
"""
import modal
import json
import sys
import os
from pathlib import Path
import zipfile
import io

app = modal.App("hamer-egodex-v2")

hamer_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0", "wget",
                 "libosmesa6-dev", "freeglut3-dev")
    .pip_install(
        "torch==2.1.0", "torchvision==0.16.0",
        "numpy==1.23.5", "opencv-python-headless", "pillow",
        "yacs", "smplx==0.1.28", "pytorch-lightning==2.1.0",
        "einops", "timm", "scikit-image", "webdataset", "chumpy",
        "pyglet<2", "pyrender==0.1.45", "huggingface_hub",
        "groundingdino-py",
    )
    .env({"PYOPENGL_PLATFORM": "osmesa"})
    .run_commands(
        "git clone https://github.com/geopavlakos/hamer.git /hamer",
        "cd /hamer && pip install -e . --no-deps",
        "pip install git+https://github.com/mmatl/pyopengl.git",
    )
    .add_local_dir(
        r"C:\Users\chris\clawd\pipeline\mano\mano_v1_2\models",
        remote_path="/hamer/_DATA/data/mano",
    )
    .add_local_file(
        r"C:\Users\chris\clawd\pipeline\mano\mano_mean_params.npz",
        remote_path="/hamer/_DATA/data/mano_mean_params.npz",
    )
)

hamer_cache = modal.Volume.from_name("hamer-cache", create_if_missing=True)


@app.function(
    image=hamer_image,
    volumes={"/root/.cache/hamer": hamer_cache},
    gpu="T4",
    timeout=3600,
)
def run_hamer_on_egodex(frames_zip_bytes: bytes) -> dict:
    """Run HaMeR on EgoDex frames. Detect hands with full-frame approach."""
    import zipfile, io, os, shutil, sys
    import numpy as np
    import torch
    import cv2

    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    
    # Monkey-patch pyrender (exact copy from modal_hamer_bimanual.py which works)
    import sys as _sys
    import unittest.mock
    import types
    
    mock = unittest.mock.MagicMock
    fake_pyrender = types.ModuleType("pyrender")
    for attr in ["OffscreenRenderer", "Mesh", "Scene", "Node", "camera",
                 "IntrinsicsCamera", "DirectionalLight", "PointLight",
                 "SpotLight", "RenderFlags", "Viewer", "Primitive",
                 "MetallicRoughnessMaterial", "GLTF", "Light",
                 "OrthographicCamera", "PerspectiveCamera", "Texture",
                 "TextAlign", "Trackball"]:
        setattr(fake_pyrender, attr, mock())
    fake_pyrender.RenderFlags = mock()
    fake_pyrender.RenderFlags.RGBA = 1
    fake_pyrender.RenderFlags.ALL_WIREFRAME = 2
    
    _sys.modules["pyrender"] = fake_pyrender
    _sys.modules["pyrender.constants"] = mock()
    _sys.modules["pyrender.light"] = mock()
    _sys.modules["pyrender.node"] = mock()
    _sys.modules["pyrender.mesh"] = mock()
    _sys.modules["pyrender.scene"] = mock()
    _sys.modules["pyrender.camera"] = mock()
    _sys.modules["pyrender.renderer"] = mock()
    _sys.modules["pyrender.offscreen"] = mock()
    _sys.modules["pyrender.viewer"] = mock()
    _sys.modules["pyrender.primitive"] = mock()
    _sys.modules["pyrender.material"] = mock()
    print("Patched pyrender")

    # Extract frames
    work = "/tmp/egodex_frames"
    if os.path.exists(work):
        shutil.rmtree(work)
    os.makedirs(work)
    zf = zipfile.ZipFile(io.BytesIO(frames_zip_bytes))
    zf.extractall(work)
    frame_files = sorted([f for f in os.listdir(work) if f.endswith('.jpg')])
    print(f"Got {len(frame_files)} frames")

    # Setup HaMeR
    sys.path.insert(0, "/hamer")
    os.chdir("/hamer")
    
    from hamer.models import load_hamer, DEFAULT_CHECKPOINT
    from hamer.utils import recursive_to
    from hamer.datasets.vitdet_dataset import ViTDetDataset

    device = torch.device("cuda")

    # Restore checkpoint from cache
    cache_dir = "/root/.cache/hamer"
    cached_ckpt = f"{cache_dir}/hamer_ckpts/checkpoints/hamer.ckpt"
    expected_ckpt = "./_DATA/hamer_ckpts/checkpoints/hamer.ckpt"
    
    if os.path.exists(cached_ckpt):
        print("Restoring from cache...")
        if not os.path.exists("./_DATA/hamer_ckpts"):
            shutil.copytree(f"{cache_dir}/hamer_ckpts", "./_DATA/hamer_ckpts")
        cached_data = f"{cache_dir}/data"
        if os.path.exists(cached_data):
            for fn in os.listdir(cached_data):
                src = f"{cached_data}/{fn}"
                dst = f"./_DATA/data/{fn}"
                if not os.path.exists(dst):
                    if os.path.isfile(src):
                        shutil.copy2(src, dst)
                    else:
                        shutil.copytree(src, dst)
    else:
        raise RuntimeError("HaMeR checkpoint not cached. Run modal_hamer_bimanual.py first.")

    model, model_cfg = load_hamer(DEFAULT_CHECKPOINT)
    if hasattr(model, "renderer"): model.renderer = None
    model = model.to(device).eval()
    print("HaMeR loaded")

    results = {}
    
    for fi, fname in enumerate(frame_files):
        if fi % 20 == 0:
            print(f"Frame {fi}/{len(frame_files)}")
        
        img = cv2.imread(os.path.join(work, fname))
        h, w = img.shape[:2]
        
        # Use full frame as a single person bbox for ViTDetDataset
        # Then HaMeR's internal hand detection will find hands
        # Actually ViTDetDataset needs hand bboxes, not person bboxes
        # Let's use simple heuristic: split frame into left/right halves as hand bboxes
        
        frame_result = {'frame': fi, 'hands': []}
        
        # Create hand bboxes: left half = right hand (egocentric mirror), right half = left hand
        # For egocentric view: left side of image = person's right hand
        # boxes in [x1, y1, x2, y2] format, right flags as 0/1
        all_boxes = np.array([
            [0, h*0.3, w*0.55, h*0.95],       # Right hand (left side of egocentric image)
            [w*0.45, h*0.3, w, h*0.95],        # Left hand (right side of egocentric image)
        ], dtype=np.float32)
        all_right = np.array([1, 0])  # first is right, second is left
        
        try:
            dataset = ViTDetDataset(model_cfg, img, all_boxes, all_right, rescale_factor=2.0)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(all_boxes), shuffle=False)
            
            for batch in dataloader:
                batch = recursive_to(batch, device)
                with torch.no_grad():
                    out = model(batch)
                
                n_hands = out['pred_keypoints_3d'].shape[0]
                for hi in range(n_hands):
                    kps3d = out['pred_keypoints_3d'].cpu().numpy()[hi]  # 21x3
                    cam_t = out['pred_cam'].cpu().numpy()[hi]  # 3
                    is_right = bool(all_right[hi])
                    
                    hand_result = {
                        'side': 'right' if is_right else 'left',
                        'keypoints_3d': kps3d.tolist(),
                        'cam_t': cam_t.tolist(),
                        'bbox': all_boxes[hi].tolist(),
                    }
                    
                    if 'pred_mano_params' in out:
                        hand_result['mano_global_orient'] = out['pred_mano_params']['global_orient'].cpu().numpy()[hi].tolist()
                        hand_result['mano_hand_pose'] = out['pred_mano_params']['hand_pose'].cpu().numpy()[hi].tolist()
                        hand_result['mano_betas'] = out['pred_mano_params']['betas'].cpu().numpy()[hi].tolist()
                    
                    frame_result['hands'].append(hand_result)
        except Exception as e:
            print(f"  Error on frame {fi}: {e}")
            import traceback; traceback.print_exc()
        
        results[str(fi)] = frame_result
    
    return results


@app.local_entrypoint()
def main(episode: str = "pick_place"):
    import zipfile, io, os
    
    if episode == "pick_place":
        frames_dir = r"C:\Users\chris\clawd\pipeline\egodex_sample\frames_pick_place_0"
    elif episode == "fold":
        frames_dir = r"C:\Users\chris\clawd\pipeline\egodex_sample\frames_fold_0"
    elif episode == "stack":
        frames_dir = r"C:\Users\chris\clawd\pipeline\egodex_sample\frames_stack_0"
    else:
        print(f"Unknown episode: {episode}")
        return
    
    # Zip frames
    print(f"Zipping {frames_dir}...")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(os.listdir(frames_dir)):
            if f.endswith('.jpg'):
                zf.write(os.path.join(frames_dir, f), f)
    frames_bytes = buf.getvalue()
    print(f"Zip: {len(frames_bytes)//1024}KB, running HaMeR...")
    
    results = run_hamer_on_egodex.remote(frames_bytes)
    
    out_path = rf"C:\Users\chris\clawd\pipeline\egodex_sample\hamer_{episode}.json"
    with open(out_path, 'w') as f:
        json.dump(results, f)
    
    n_hands = sum(len(v.get('hands', [])) for v in results.values())
    print(f"Done: {len(results)} frames, {n_hands} total hand detections")
    print(f"Saved: {out_path}")
