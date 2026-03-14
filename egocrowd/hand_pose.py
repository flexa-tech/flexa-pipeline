"""3D hand mesh reconstruction via HaMeR on Modal GPU.

Calls the Modal function defined in processing/hamer_modal.py.
Falls back gracefully if Modal is unavailable (caller catches exception).
"""

from typing import Optional


def extract_hand_poses(
    rgb_dir: str,
    hand_boxes: Optional[dict] = None,
    device: Optional[str] = None,
) -> dict:
    """Extract 3D hand meshes from RGB frames using HaMeR on Modal GPU.

    Calls the Modal function defined in processing/hamer_modal.py.
    Falls back gracefully if Modal is unavailable.

    Args:
        rgb_dir: Directory containing RGB frames (.jpg)
        hand_boxes: Optional pre-computed hand bounding boxes (unused, HaMeR detects internally)
        device: Ignored (runs on Modal GPU)

    Returns:
        dict with per-frame MANO parameters, joint positions, and translations.
        Format: {"timesteps": [...], "detection_rate": float, "model": "hamer"|"gdino-only"}
    """
    from pathlib import Path

    rgb_path = Path(rgb_dir)
    frames = sorted(rgb_path.glob("*.jpg"))
    if not frames:
        raise FileNotFoundError(f"No .jpg frames found in {rgb_dir}")

    print(f"Processing {len(frames)} frames with HaMeR (Modal GPU)...")

    # Load frame bytes
    frame_bytes = []
    for f in frames:
        frame_bytes.append(f.read_bytes())

    # Call Modal function
    import modal
    run_hamer = modal.Function.lookup("flexa-hamer", "run_hamer_inference")
    result = run_hamer.remote(frame_bytes)

    detection_rate = result.get("detection_rate", 0)
    n_detected = sum(1 for ts in result.get("timesteps", []) if ts.get("detected"))
    print(f"HaMeR detection: {n_detected}/{len(frames)} frames ({detection_rate:.1%})")

    return result
