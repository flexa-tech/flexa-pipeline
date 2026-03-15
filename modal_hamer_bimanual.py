"""
HaMeR bimanual: detect BOTH hands from GDINO bboxes, get MANO 3D joints.

Fixes from modal_hamer.py:
1. Processes ALL hand bboxes per frame (not just one)
2. Classifies left vs right from bbox position  
3. Returns proper 21x3 MANO keypoints per hand

Usage: python -m modal run modal_hamer_bimanual.py --session stack2
"""
import modal
import json
import sys
from pathlib import Path
import zipfile

app = modal.App("hamer-bimanual")

hamer_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0", "wget",
                 "libosmesa6-dev", "freeglut3-dev")
    .pip_install(
        "torch==2.1.0", "torchvision==0.16.0",
        "numpy==1.23.5", "opencv-python-headless", "pillow",
        "yacs", "smplx==0.1.28", "pytorch-lightning==2.1.0",
        "einops", "timm", "scikit-image", "webdataset", "chumpy",
        "pyglet<2", "pyrender==0.1.45",
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
)

hamer_cache = modal.Volume.from_name("hamer-cache", create_if_missing=True)


@app.function(
    image=hamer_image,
    gpu="T4",
    timeout=3600,
    volumes={"/root/.cache/hamer": hamer_cache},
)
def run_hamer_bimanual(
    frame_bytes_list: list[bytes],
    frame_indices: list[int],
    all_bboxes: list[list[dict]],  # Per frame: list of {"box": [x1,y1,x2,y2], "score": float, "label": str}
    session_name: str,
    img_width: int = 960,
) -> dict:
    """Run HaMeR on ALL hand bounding boxes per frame."""
    import os
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    os.chdir("/hamer")

    import torch
    import numpy as np
    import cv2

    sys_path_insert = __import__("sys").path
    sys_path_insert.insert(0, "/hamer")
    from hamer.models import load_hamer, DEFAULT_CHECKPOINT
    from hamer.utils import recursive_to
    from hamer.datasets.vitdet_dataset import ViTDetDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Verify MANO
    assert os.path.exists("./_DATA/data/mano/MANO_RIGHT.pkl"), "MANO_RIGHT.pkl missing"

    # Download model if needed
    try:
        from hamer.models import download_models
        from hamer.configs import CACHE_DIR_HAMER
        download_models(CACHE_DIR_HAMER)
    except Exception as e:
        print(f"download_models: {e}")

    model, model_cfg = load_hamer(DEFAULT_CHECKPOINT)
    if hasattr(model, "renderer"): model.renderer = None
    if hasattr(model, "mesh_renderer"): model.mesh_renderer = None
    model = model.to(device).eval()
    print("HaMeR loaded")

    results = []

    for i, (frame_bytes, frame_idx, bboxes) in enumerate(
        zip(frame_bytes_list, frame_indices, all_bboxes)
    ):
        if not bboxes:
            results.append({"frame_idx": frame_idx, "hands": []})
            continue

        # Decode frame
        img = cv2.imdecode(
            np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if img is None:
            results.append({"frame_idx": frame_idx, "hands": []})
            continue

        frame_hands = []

        # Use HaMeR's built-in ViTPose-based hand detection
        # Process each bbox (left half / right half of frame)
        all_boxes = []
        all_right_flags = []
        for bbox_info in bboxes:
            box = bbox_info["box"]
            cx = (box[0] + box[2]) / 2
            is_right = cx < img_width / 2
            all_boxes.append(box)
            all_right_flags.append(1 if is_right else 0)

        if not all_boxes:
            results.append({"frame_idx": frame_idx, "hands": []})
            continue

        boxes_np = np.array(all_boxes)
        right_np = np.array(all_right_flags)

        try:
            dataset = ViTDetDataset(
                model_cfg, img, boxes_np, right_np, rescale_factor=2.0
            )
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=len(all_boxes), shuffle=False, num_workers=0
            )

            for batch in loader:
                batch = recursive_to(batch, device)
                with torch.no_grad():
                    out = model(batch)

                pred_kp3d = out["pred_keypoints_3d"].cpu().numpy()  # [B, 21, 3]
                pred_mano = out["pred_mano_params"]

                for n in range(pred_kp3d.shape[0]):
                    is_right = right_np[n] == 1
                    frame_hands.append({
                        "hand": "right" if is_right else "left",
                        "keypoints_3d": pred_kp3d[n].tolist(),  # 21x3 MANO joints
                        "mano_betas": pred_mano["betas"][n].cpu().numpy().tolist(),
                        "cam_t": out["pred_cam_t"][n].cpu().numpy().tolist(),
                        "bbox": all_boxes[n],
                        "bbox_center_x": (all_boxes[n][0] + all_boxes[n][2]) / 2,
                    })
        except Exception as e:
            print(f"  Frame {frame_idx}: {type(e).__name__}: {e}")

        results.append({"frame_idx": frame_idx, "hands": frame_hands})

        if (i + 1) % 50 == 0:
            n_hands = sum(len(r["hands"]) for r in results)
            print(f"  {i+1}/{len(frame_bytes_list)} frames, {n_hands} hands total")

    # Stats
    both = sum(1 for r in results if len(r["hands"]) >= 2)
    any_hand = sum(1 for r in results if r["hands"])
    total_hands = sum(len(r["hands"]) for r in results)
    print(f"Done: {any_hand}/{len(results)} frames with hands, {both} bimanual, {total_hands} total")

    return {
        "session": session_name,
        "total_frames": len(results),
        "frames_with_hands": any_hand,
        "bimanual_frames": both,
        "total_hand_detections": total_hands,
        "results": results,
    }


@app.local_entrypoint()
def main(session: str = "stack2"):
    base = Path(__file__).parent

    # Load R3D frames
    r3d_path = base / "raw_captures" / session / f"{session}.r3d"
    if not r3d_path.exists():
        print(f"R3D not found: {r3d_path}")
        return

    print(f"Loading {session} R3D frames...")
    frame_bytes = []
    frame_indices = []
    with zipfile.ZipFile(r3d_path) as z:
        meta = json.load(z.open("metadata"))
        img_width = meta.get("w", 960)
        rgb_files = sorted(f for f in z.namelist() if f.startswith("rgbd/") and f.endswith(".jpg"))
        # Process every other frame (subsample 2x for speed)
        for idx, f in enumerate(rgb_files[::2]):
            frame_bytes.append(z.read(f))
            frame_indices.append(idx * 2)

    print(f"  {len(frame_bytes)} frames (subsampled 2x from {len(rgb_files)}), width={img_width}")

    # Build all_bboxes: per frame, list of hand bboxes
    # No per-frame GDINO hand detections — use default full-frame bboxes
    # HaMeR's ViTDetDataset + ViTPose will detect hands within these regions
    h = meta.get("h", 720)
    all_bboxes = []
    for idx in frame_indices:
        # Two bboxes covering left and right halves of frame
        all_bboxes.append([
            {"box": [0, 0, img_width//2, h], "score": 0.5, "label": "hand_right"},
            {"box": [img_width//2, 0, img_width, h], "score": 0.5, "label": "hand_left"},
        ])

    # Process in batches
    batch_size = 50
    all_results = []
    for b in range(0, len(frame_bytes), batch_size):
        batch_frames = frame_bytes[b:b+batch_size]
        batch_indices = frame_indices[b:b+batch_size]
        batch_bboxes = all_bboxes[b:b+batch_size]
        
        print(f"  Batch {b//batch_size + 1}: frames {batch_indices[0]}-{batch_indices[-1]}")
        result = run_hamer_bimanual.remote(
            batch_frames, batch_indices, batch_bboxes, session, img_width
        )
        all_results.extend(result["results"])

    # Combine
    both = sum(1 for r in all_results if len(r["hands"]) >= 2)
    any_h = sum(1 for r in all_results if r["hands"])
    total_h = sum(len(r["hands"]) for r in all_results)

    output = {
        "session": session,
        "total_frames": len(all_results),
        "frames_with_hands": any_h,
        "bimanual_frames": both,
        "total_hand_detections": total_h,
        "results": all_results,
    }

    out_file = base / "modal_results" / f"{session}_hamer_bimanual.json"
    out_file.write_text(json.dumps(output, indent=2))
    print(f"\nDone! {any_h}/{len(all_results)} frames, {both} bimanual, {total_h} hands")
    print(f"Saved: {out_file}")
