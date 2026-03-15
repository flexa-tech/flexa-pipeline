"""
LeRobot (our format) → LingBot-VLA Compatible Dataset Converter

Bridges the gap between our Flexa_to_lerobot.py output and what LingBot-VLA
expects for post-training/fine-tuning.

Gaps addressed:
1. Image key: observation.image → observation.images.cam_high (+ dummy wrist cams)
2. Storage format: .npy + loose JPGs → proper LeRobot v2 HuggingFace dataset
3. Normalization stats: generates the norm_stats JSON LingBot requires
4. State/action dim padding: pads to LingBot's max_state_dim / max_action_dim

Usage:
    python lerobot_to_lingbot.py --input /data/lerobot_output --output /data/lingbot_ready
    python lerobot_to_lingbot.py --input /data/lerobot_output --output /data/lingbot_ready --compute-norm-stats
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import torch
    from datasets import Dataset, Features, Sequence, Value, Image as HFImage
    HAS_HF = True
except ImportError:
    HAS_HF = False


def load_our_dataset(input_dir: Path):
    """Load the .npy + images output from Flexa_to_lerobot.py"""
    input_dir = Path(input_dir)

    states = np.load(input_dir / "observation_state.npy")       # (N, state_dim)
    actions = np.load(input_dir / "action.npy")                  # (N, action_dim)
    episode_indices = np.load(input_dir / "episode_index.npy")   # (N,)
    frame_indices = np.load(input_dir / "frame_index.npy")       # (N,)
    timestamps = np.load(input_dir / "timestamp.npy")            # (N,)

    with open(input_dir / "metadata.json") as f:
        metadata = json.load(f)

    image_dir = input_dir / "images"
    image_paths = [image_dir / f"frame_{i:08d}.jpg" for i in range(len(states))]

    return {
        "states": states,
        "actions": actions,
        "episode_indices": episode_indices,
        "frame_indices": frame_indices,
        "timestamps": timestamps,
        "metadata": metadata,
        "image_paths": image_paths,
    }


def pad_to_dim(arr: np.ndarray, target_dim: int) -> np.ndarray:
    """Pad or truncate array's last dimension to target_dim."""
    current_dim = arr.shape[-1]
    if current_dim == target_dim:
        return arr
    if current_dim > target_dim:
        return arr[..., :target_dim]
    padding = np.zeros((*arr.shape[:-1], target_dim - current_dim), dtype=arr.dtype)
    return np.concatenate([arr, padding], axis=-1)


def compute_normalization_stats(states, actions):
    """Compute normalization stats in LingBot format (bounds_99_woclip)."""
    def percentile_bounds(data):
        lo = np.percentile(data, 0.5, axis=0).tolist()
        hi = np.percentile(data, 99.5, axis=0).tolist()
        mean = np.mean(data, axis=0).tolist()
        std = np.std(data, axis=0).tolist()
        return {"min": lo, "max": hi, "mean": mean, "std": std}

    return {
        "action": percentile_bounds(actions),
        "observation.state": percentile_bounds(states),
    }


def convert_to_lingbot(input_dir, output_dir, max_action_dim=75, max_state_dim=75,
                        compute_norm=False, image_size=256):
    """
    Convert our LeRobot-ish output into LingBot-VLA compatible format.

    LingBot-VLA expects a LeRobot dataset with keys:
        - action: (max_action_dim,) float32
        - observation.state: (max_state_dim,) float32
        - observation.images.cam_high: PIL Image (primary view)
        - observation.images.cam_left_wrist: PIL Image (can be dummy)
        - observation.images.cam_right_wrist: PIL Image (can be dummy)
        - episode_index: int
        - frame_index: int
        - timestamp: float
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {input_dir}...")
    data = load_our_dataset(input_dir)

    n_frames = len(data["states"])
    state_dim = data["states"].shape[1]
    action_dim = data["actions"].shape[1]

    print(f"  Frames: {n_frames}")
    print(f"  State dim: {state_dim} → padded to {max_state_dim}")
    print(f"  Action dim: {action_dim} → padded to {max_action_dim}")

    # Pad state/action dims
    padded_states = pad_to_dim(data["states"], max_state_dim)
    padded_actions = pad_to_dim(data["actions"], max_action_dim)

    # Compute norm stats if requested
    if compute_norm:
        print("Computing normalization statistics...")
        norm_stats = compute_normalization_stats(padded_states, padded_actions)
        norm_path = output_dir / "norm_stats.json"
        with open(norm_path, "w") as f:
            json.dump(norm_stats, f, indent=2)
        print(f"  Saved to {norm_path}")

    # Save in LeRobot-compatible structure
    # Option A: Save as HuggingFace Dataset (if available)
    if HAS_HF:
        print("Building HuggingFace Dataset...")
        records = []
        img_out_dir = output_dir / "images"
        for cam in ["cam_high", "cam_left_wrist", "cam_right_wrist"]:
            (img_out_dir / cam).mkdir(parents=True, exist_ok=True)

        for i in tqdm(range(n_frames), desc="Processing frames"):
            ep_idx = int(data["episode_indices"][i])
            fr_idx = int(data["frame_indices"][i])

            # Copy/link primary image as cam_high
            src_img = data["image_paths"][i]
            cam_high_path = img_out_dir / "cam_high" / f"ep{ep_idx:04d}_fr{fr_idx:06d}.jpg"

            if src_img.exists():
                img = Image.open(src_img).resize((image_size, image_size))
                img.save(cam_high_path, quality=85)

                # Create dummy wrist views (black images for now — or could duplicate)
                for cam in ["cam_left_wrist", "cam_right_wrist"]:
                    dummy_path = img_out_dir / cam / f"ep{ep_idx:04d}_fr{fr_idx:06d}.jpg"
                    # Use the same egocentric image as a placeholder
                    img.save(dummy_path, quality=85)

            records.append({
                "action": padded_actions[i].tolist(),
                "observation.state": padded_states[i].tolist(),
                "observation.images.cam_high": str(cam_high_path),
                "observation.images.cam_left_wrist": str(img_out_dir / "cam_left_wrist" / f"ep{ep_idx:04d}_fr{fr_idx:06d}.jpg"),
                "observation.images.cam_right_wrist": str(img_out_dir / "cam_right_wrist" / f"ep{ep_idx:04d}_fr{fr_idx:06d}.jpg"),
                "episode_index": ep_idx,
                "frame_index": fr_idx,
                "timestamp": float(data["timestamps"][i]),
            })

        dataset = Dataset.from_list(records)
        dataset.save_to_disk(str(output_dir / "dataset"))
        print(f"HuggingFace Dataset saved to {output_dir / 'dataset'}")

    else:
        # Option B: Save as .npy files with correct naming
        print("HuggingFace datasets not available, saving as .npy...")
        np.save(output_dir / "action.npy", padded_actions)
        np.save(output_dir / "observation_state.npy", padded_states)
        np.save(output_dir / "episode_index.npy", data["episode_indices"])
        np.save(output_dir / "frame_index.npy", data["frame_indices"])
        np.save(output_dir / "timestamp.npy", data["timestamps"])

        # Reorganize images into multi-cam structure
        img_out_dir = output_dir / "images"
        for cam in ["cam_high", "cam_left_wrist", "cam_right_wrist"]:
            (img_out_dir / cam).mkdir(parents=True, exist_ok=True)

        for i in tqdm(range(n_frames), desc="Copying images"):
            src = data["image_paths"][i]
            if src.exists():
                ep_idx = int(data["episode_indices"][i])
                fr_idx = int(data["frame_indices"][i])
                fname = f"ep{ep_idx:04d}_fr{fr_idx:06d}.jpg"
                # Copy as all three views (primary + dummy wrist)
                img = Image.open(src)
                for cam in ["cam_high", "cam_left_wrist", "cam_right_wrist"]:
                    img.save(img_out_dir / cam / fname, quality=85)

    # Save updated metadata
    meta = data["metadata"].copy()
    meta["lingbot_compatible"] = True
    meta["original_state_dim"] = state_dim
    meta["original_action_dim"] = action_dim
    meta["padded_state_dim"] = max_state_dim
    meta["padded_action_dim"] = max_action_dim
    meta["image_views"] = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    meta["note"] = (
        "Converted from Flexa egocentric hand data. cam_high = egocentric view. "
        "Wrist cams are duplicates of egocentric view (no real wrist cameras in source data). "
        "State is flattened hand joint poses (14 joints × 9 = 126 dims, zero-padded to 75). "
        "Action is delta state between consecutive frames."
    )

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ LingBot-VLA compatible dataset saved to {output_dir}")
    print(f"  Use --data.train_path {output_dir} in LingBot-VLA training config")
    print(f"  Set action_dim: {min(action_dim, max_action_dim)} in training YAML")
    print(f"  Set norm_stats_file: {output_dir / 'norm_stats.json'}" if compute_norm else "")


def main():
    parser = argparse.ArgumentParser(description="Convert our LeRobot output to LingBot-VLA format")
    parser.add_argument("--input", required=True, help="Path to Flexa_to_lerobot.py output dir")
    parser.add_argument("--output", required=True, help="Output directory for LingBot-VLA format")
    parser.add_argument("--max-action-dim", type=int, default=75, help="LingBot max action dim")
    parser.add_argument("--max-state-dim", type=int, default=75, help="LingBot max state dim")
    parser.add_argument("--compute-norm-stats", action="store_true", help="Compute normalization stats")
    parser.add_argument("--image-size", type=int, default=256, help="Output image size")
    args = parser.parse_args()

    convert_to_lingbot(
        args.input, args.output,
        max_action_dim=args.max_action_dim,
        max_state_dim=args.max_state_dim,
        compute_norm=args.compute_norm_stats,
        image_size=args.image_size,
    )


if __name__ == "__main__":
    main()
