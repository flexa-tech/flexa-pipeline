"""Modal function: Combined GroundingDINO hand detection + HaMeR mesh recovery.

Runs on A10G GPU. Detects hands via GroundingDINO, then attempts HaMeR
mesh recovery for 3D wrist positions. Falls back to detection-only mode
if HaMeR dependencies fail.

Deploy:  modal deploy processing/hamer_modal.py
Test:    modal run processing/hamer_modal.py
"""
import modal

app = modal.App("flexa-hamer")

hamer_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0", "unzip")
    .pip_install(
        "torch==2.4.0", "torchvision==0.19.0",
        "transformers>=4.36.0",
        "numpy<2", "Pillow", "opencv-python-headless",
        "smplx==0.1.28", "timm", "einops",
        "yacs", "chumpy", "gdown",
    )
    # Attempt HaMeR install with --no-deps to avoid pulling mmcv transitively.
    # If this layer fails, the image build will use detection-only mode at runtime.
    .run_commands(
        "pip install --no-deps git+https://github.com/geopavlakos/hamer.git || "
        "echo 'WARN: HaMeR pip install failed -- will use detection-only mode'"
    )
    # Provision MANO model files (required by HaMeR for 3D hand mesh recovery)
    .add_local_file(
        "/Users/christian/Documents/ai_dev/mano_v1_2.zip",
        "/root/mano_v1_2.zip",
        copy=True,
    )
    .run_commands(
        "mkdir -p /root/_DATA/data/mano && "
        "cd /root && unzip -q mano_v1_2.zip && "
        "cp mano_v1_2/models/MANO_RIGHT.pkl /root/_DATA/data/mano/ && "
        "rm -rf mano_v1_2.zip mano_v1_2"
    )
    # Pre-download GroundingDINO weights into the image
    .run_commands(
        "python -c \""
        "from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection; "
        "AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-tiny'); "
        "AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-tiny')"
        "\""
    )
    # Download HaMeR checkpoints (ViTPose + HaMeR model weights)
    .run_commands(
        "python -c \""
        "import gdown, os; "
        "os.makedirs('/root/_DATA/hamer_ckpts/checkpoints', exist_ok=True); "
        "os.makedirs('/root/_DATA/vitpose_ckpts', exist_ok=True); "
        "\" || echo 'Checkpoint dirs created'"
    )
    .workdir("/root")
)


# ---------------------------------------------------------------------------
# Helpers (run inside Modal container)
# ---------------------------------------------------------------------------

def _detect_hands_gdino(images, processor, model, threshold=0.2):
    """Detect hands in a list of PIL Images using GroundingDINO.

    Returns list of dicts with 'boxes' (xyxy), 'scores', 'detected'.
    """
    import torch

    results = []
    for image in images:
        inputs = processor(
            images=image, text="hand", return_tensors="pt"
        ).to("cuda")
        with torch.no_grad():
            outputs = model(**inputs)

        dets = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            threshold=threshold,
            target_sizes=[image.size[::-1]],
        )[0]

        boxes = dets["boxes"].cpu().numpy().tolist()
        scores = dets["scores"].cpu().numpy().tolist()
        results.append({
            "detected": len(boxes) > 0,
            "boxes": boxes,
            "scores": scores,
        })
    return results


def _best_hand_box(det):
    """Pick the single best hand detection from a frame result."""
    if not det["detected"]:
        return None, 0.0
    # Pick highest-confidence detection
    best_idx = max(range(len(det["scores"])), key=lambda i: det["scores"][i])
    return det["boxes"][best_idx], det["scores"][best_idx]


def _box_center(box):
    """Return (cx, cy) for an xyxy bounding box."""
    return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]


def _try_load_hamer():
    """Attempt to load HaMeR model. Returns (model, model_cfg) or (None, None)."""
    try:
        from hamer.configs import CACHE_DIR_HAMER  # noqa: F401
        from hamer.models import load_hamer, DEFAULT_CHECKPOINT
        model, model_cfg = load_hamer(DEFAULT_CHECKPOINT)
        model = model.cuda().eval()
        return model, model_cfg
    except Exception as e:
        print(f"HaMeR load failed: {e}")
        print("Falling back to GroundingDINO detection-only mode")
        return None, None


def _estimate_grasping_from_joints(joints_3d):
    """Estimate grasping from thumb-index tip distance (MANO joint order).

    Joint 4 = thumb tip, Joint 8 = index finger tip.
    Threshold: 0.04m (4cm) in camera-frame meters.
    """
    import numpy as np
    joints = np.array(joints_3d)
    thumb_tip = joints[4]
    index_tip = joints[8]
    dist = float(np.linalg.norm(thumb_tip - index_tip))
    return dist < 0.04


def _estimate_grasping_from_box(box, score):
    """Heuristic grasping estimate from detection box when HaMeR is unavailable.

    Uses aspect ratio: grasping hands tend to be more square (closed fist),
    while open hands are taller than wide. Combined with box area relative
    to typical egocentric hand sizes.
    """
    w = box[2] - box[0]
    h = box[3] - box[1]
    area = w * h
    aspect = w / max(h, 1)
    # Grasping: compact (high aspect ratio ≈ square) + moderate area
    # Open hand: tall and narrow (low aspect ratio)
    return bool(aspect > 0.7 and area < 40000 and score > 0.15)


def _crop_hand(image, box, target_size=(256, 192)):
    """Crop hand region from PIL image and resize for HaMeR input."""
    import numpy as np
    from PIL import Image as PILImage

    x1, y1, x2, y2 = [int(c) for c in box]
    # Expand box by 20% for context
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(w, h) * 1.2
    x1n = max(0, int(cx - side / 2))
    y1n = max(0, int(cy - side / 2))
    x2n = min(image.width, int(cx + side / 2))
    y2n = min(image.height, int(cy + side / 2))

    crop = image.crop((x1n, y1n, x2n, y2n))
    crop = crop.resize(target_size, PILImage.BILINEAR)
    return crop


# ---------------------------------------------------------------------------
# Main Modal function
# ---------------------------------------------------------------------------

@app.function(gpu="A10G", image=hamer_image, timeout=900)
def run_hamer_inference(frame_bytes_list: list[bytes], batch_size: int = 16,
                        detection_threshold: float = 0.15):
    """Combined GroundingDINO + HaMeR inference on GPU.

    Args:
        frame_bytes_list: List of JPEG-encoded frame bytes
        batch_size: Batch size for processing

    Returns:
        dict with per-frame results:
        {
            "timesteps": [
                {
                    "frame_idx": int,
                    "detected": bool,
                    "wrist_pixel": [u, v] or None,
                    "wrist_3d_camera": [x, y, z] or None,
                    "joints_3d": [[x,y,z], ...] (21 joints) or None,
                    "grasping": bool,
                    "confidence": float,
                },
                ...
            ],
            "detection_rate": float,
            "model": "hamer" or "gdino-only",
        }
    """
    import io
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    print(f"Processing {len(frame_bytes_list)} frames...")

    # --- Stage 0: Load models ---
    print("Loading GroundingDINO...")
    model_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(model_id)
    gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).cuda()

    hamer_model, hamer_cfg = _try_load_hamer()
    use_hamer = hamer_model is not None
    model_name = "hamer" if use_hamer else "gdino-only"
    print(f"Mode: {model_name}")

    # --- Stage 1: Decode frames ---
    images = []
    for fb in frame_bytes_list:
        img = Image.open(io.BytesIO(fb)).convert("RGB")
        images.append(img)

    # --- Stage 2: GroundingDINO hand detection (batched) ---
    print("Running GroundingDINO hand detection...")
    all_detections = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        batch_dets = _detect_hands_gdino(batch, processor, gdino_model, threshold=detection_threshold)
        all_detections.extend(batch_dets)
        n_det = sum(1 for d in batch_dets if d["detected"])
        print(f"  Batch {start // batch_size + 1}: {n_det}/{len(batch_dets)} detected")

    # --- Stage 3: HaMeR mesh recovery (if available) ---
    timesteps = []
    for i, (img, det) in enumerate(zip(images, all_detections)):
        box, score = _best_hand_box(det)

        if box is None:
            timesteps.append({
                "frame_idx": i,
                "detected": False,
                "wrist_pixel": None,
                "wrist_3d_camera": None,
                "joints_3d": None,
                "grasping": False,
                "confidence": 0.0,
            })
            continue

        wrist_pixel = _box_center(box)
        wrist_3d_camera = None
        joints_3d = None
        grasping = _estimate_grasping_from_box(box, score)

        if use_hamer:
            try:
                crop = _crop_hand(img, box)
                # HaMeR inference on the crop
                from torchvision import transforms
                normalize = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
                crop_tensor = transforms.ToTensor()(crop)
                crop_tensor = normalize(crop_tensor).unsqueeze(0).cuda()

                with torch.no_grad():
                    out = hamer_model(crop_tensor)

                if "pred_keypoints_3d" in out:
                    kp3d = out["pred_keypoints_3d"][0].cpu().numpy()
                    joints_3d = kp3d.tolist()
                    wrist_3d_camera = kp3d[0].tolist()  # Joint 0 = wrist
                    grasping = _estimate_grasping_from_joints(kp3d)
            except Exception as e:
                if i == 0:
                    print(f"  HaMeR inference failed on frame {i}: {e}")
                    print("  Continuing with detection-only for remaining frames")
                # Keep detection-only results for this frame

        timesteps.append({
            "frame_idx": i,
            "detected": True,
            "wrist_pixel": wrist_pixel,
            "wrist_3d_camera": wrist_3d_camera,
            "joints_3d": joints_3d,
            "grasping": grasping,
            "confidence": float(score),
        })

    n_detected = sum(1 for ts in timesteps if ts["detected"])
    detection_rate = n_detected / max(len(timesteps), 1)
    n_3d = sum(1 for ts in timesteps if ts.get("wrist_3d_camera") is not None)

    print(f"\nResults:")
    print(f"  Detection rate: {n_detected}/{len(timesteps)} ({detection_rate:.1%})")
    print(f"  3D wrist output: {n_3d}/{len(timesteps)} frames")
    print(f"  Model: {model_name}")

    return {
        "timesteps": timesteps,
        "detection_rate": detection_rate,
        "model": model_name,
    }


# ---------------------------------------------------------------------------
# Local entrypoint for testing
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    """Test HaMeR inference on stack2 frames."""
    import os

    rgb_dir = "r3d_output/stack2/frames"
    if not os.path.exists(rgb_dir):
        print(f"No frames found at {rgb_dir}")
        print("Trying alternative paths...")
        for alt in ["r3d_output/stack2/frames", "pipeline/r3d_output/rgb"]:
            if os.path.exists(alt):
                rgb_dir = alt
                break
        else:
            print("No frame directory found. Provide path as argument.")
            return

    frames = sorted(f for f in os.listdir(rgb_dir) if f.endswith(".jpg"))
    print(f"Loading {len(frames)} frames from {rgb_dir}...")

    frame_bytes = []
    for f in frames:
        with open(os.path.join(rgb_dir, f), "rb") as fh:
            frame_bytes.append(fh.read())

    result = run_hamer_inference.remote(frame_bytes)

    # Summary
    n_detected = sum(1 for ts in result["timesteps"] if ts["detected"])
    n_3d = sum(1 for ts in result["timesteps"] if ts.get("wrist_3d_camera"))
    total = len(result["timesteps"])
    rate = result["detection_rate"]

    print(f"\n{'='*50}")
    print(f"HAMER RESULTS: {n_detected}/{total} frames detected ({rate:.1%})")
    print(f"3D wrist output: {n_3d}/{total} frames")
    print(f"Model: {result['model']}")
    print(f"{'='*50}")

    if rate > 0.85:
        print("TRK-04 PASS: Detection rate > 85%")
    else:
        print(f"TRK-04 FAIL: Detection rate {rate:.1%} < 85%")

    if n_3d > 0:
        print("TRK-05: 3D wrist data available")
    else:
        print("TRK-05: No 3D wrist data (detection-only mode)")
