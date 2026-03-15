"""
Flexa → LeRobot Format Converter
Converts Apple Flexa egocentric manipulation data to LeRobot-compatible episodes.

Input: Flexa HDF5 + MP4 pairs (hand pose + video)
Output: LeRobot dataset (HuggingFace datasets format)

Usage:
    python Flexa_to_lerobot.py --input /data/Flexa/test --output /data/output --max-episodes 50
"""

import argparse
import os
import glob
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Key hand joints for manipulation (subset of 68)
KEY_JOINTS = [
    "leftHand", "rightHand",
    "leftIndexFingerTip", "leftIndexFingerKnuckle",
    "leftMiddleFingerTip", "leftMiddleFingerKnuckle", 
    "leftThumbTip", "leftThumbKnuckle",
    "rightIndexFingerTip", "rightIndexFingerKnuckle",
    "rightMiddleFingerTip", "rightMiddleFingerKnuckle",
    "rightThumbTip", "rightThumbKnuckle",
]


def extract_joint_state(hdf5_file, frame_idx, joints=KEY_JOINTS):
    """Extract flattened joint positions from a single frame."""
    state = []
    for joint in joints:
        key = f"transforms/{joint}"
        if key in hdf5_file:
            transform = hdf5_file[key][frame_idx]  # 4x4 SE3
            # Extract position (translation) from transform matrix
            position = transform[:3, 3]  # x, y, z
            # Extract rotation as quaternion-like (first 3 euler angles from rotation matrix)
            rotation = transform[:3, :3]
            # Use 6D rotation representation (first two columns of rotation matrix)
            rot_6d = rotation[:, :2].flatten()  # 6 values
            state.extend(position.tolist())
            state.extend(rot_6d.tolist())
        else:
            state.extend([0.0] * 9)  # 3 pos + 6 rot
    return np.array(state, dtype=np.float32)


def extract_video_frames(mp4_path, target_fps=10, max_frames=None):
    """Extract frames from MP4 at target FPS using ffmpeg."""
    import subprocess
    import tempfile
    
    # Validate inputs
    if not os.path.isfile(mp4_path):
        raise FileNotFoundError(f"Video not found: {mp4_path}")
    target_fps = int(target_fps)  # prevent injection via fps param
    
    tmpdir = tempfile.mkdtemp()
    cmd = [
        "ffmpeg", "-i", mp4_path,
        "-vf", f"fps={target_fps}",
        "-q:v", "2",
        f"{tmpdir}/frame_%06d.jpg"
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    
    frames = sorted(glob.glob(f"{tmpdir}/frame_*.jpg"))
    if max_frames:
        frames = frames[:max_frames]
    
    images = []
    for f in frames:
        img = Image.open(f).resize((256, 256))
        images.append(np.array(img))
        os.remove(f)
    os.rmdir(tmpdir)
    
    return images


def compute_actions(states):
    """Compute delta actions between consecutive states."""
    actions = []
    for i in range(len(states) - 1):
        delta = states[i + 1] - states[i]
        actions.append(delta)
    # Last action is zero (terminal)
    actions.append(np.zeros_like(actions[0]))
    return actions


def process_episode(hdf5_path, mp4_path, episode_idx, target_fps=10):
    """Process a single Flexa episode into LeRobot format."""
    with h5py.File(hdf5_path, "r") as f:
        n_frames = f["transforms/camera"].shape[0]
        source_fps = 30
        step = max(1, source_fps // target_fps)
        
        # Extract states at target FPS
        frame_indices = list(range(0, n_frames, step))
        states = []
        for idx in frame_indices:
            state = extract_joint_state(f, idx)
            states.append(state)
        
        # Get language description
        description = ""
        if "llm_description" in f.attrs:
            description = f.attrs["llm_description"]
        
        # Get confidences if available
        has_confidence = "confidences/leftHand" in f
    
    # Extract video frames
    frames = extract_video_frames(mp4_path, target_fps=target_fps, max_frames=len(states))
    
    # Align lengths
    min_len = min(len(states), len(frames))
    states = states[:min_len]
    frames = frames[:min_len]
    
    # Compute delta actions
    actions = compute_actions(np.array(states))
    
    # Package as LeRobot episode
    episode = {
        "episode_index": episode_idx,
        "task_description": description,
        "n_frames": min_len,
        "fps": target_fps,
        "frames": [],
    }
    
    for i in range(min_len):
        frame_data = {
            "observation.image": frames[i],           # H x W x 3 uint8
            "observation.state": states[i],            # flat joint state vector
            "action": actions[i],                      # delta action
            "episode_index": episode_idx,
            "frame_index": i,
            "timestamp": i / target_fps,
        }
        episode["frames"].append(frame_data)
    
    return episode


def save_lerobot_dataset(episodes, output_dir):
    """Save episodes in LeRobot-compatible format."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Flatten all frames
    all_data = {
        "observation.state": [],
        "action": [],
        "episode_index": [],
        "frame_index": [],
        "timestamp": [],
    }
    
    image_dir = output_dir / "images"
    image_dir.mkdir(exist_ok=True)
    
    metadata = {
        "fps": episodes[0]["fps"] if episodes else 10,
        "n_episodes": len(episodes),
        "state_dim": len(episodes[0]["frames"][0]["observation.state"]) if episodes else 0,
        "action_dim": len(episodes[0]["frames"][0]["action"]) if episodes else 0,
        "tasks": [],
    }
    
    frame_count = 0
    for ep in tqdm(episodes, desc="Saving episodes"):
        metadata["tasks"].append({
            "episode_index": ep["episode_index"],
            "description": ep["task_description"],
            "n_frames": ep["n_frames"],
        })
        
        for frame in ep["frames"]:
            # Save image
            img = Image.fromarray(frame["observation.image"])
            img.save(image_dir / f"frame_{frame_count:08d}.jpg", quality=85)
            
            all_data["observation.state"].append(frame["observation.state"].tolist())
            all_data["action"].append(frame["action"].tolist())
            all_data["episode_index"].append(frame["episode_index"])
            all_data["frame_index"].append(frame["frame_index"])
            all_data["timestamp"].append(frame["timestamp"])
            frame_count += 1
    
    # Save as numpy arrays
    for key in ["observation.state", "action"]:
        arr = np.array(all_data[key], dtype=np.float32)
        np.save(output_dir / f"{key.replace('.', '_')}.npy", arr)
    
    for key in ["episode_index", "frame_index"]:
        arr = np.array(all_data[key], dtype=np.int64)
        np.save(output_dir / f"{key}.npy", arr)
    
    timestamps = np.array(all_data["timestamp"], dtype=np.float64)
    np.save(output_dir / "timestamp.npy", timestamps)
    
    # Save metadata
    metadata["total_frames"] = frame_count
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nDataset saved to {output_dir}")
    print(f"  Episodes: {len(episodes)}")
    print(f"  Total frames: {frame_count}")
    print(f"  State dim: {metadata['state_dim']}")
    print(f"  Action dim: {metadata['action_dim']}")
    
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Convert Flexa to LeRobot format")
    parser.add_argument("--input", required=True, help="Path to Flexa data directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--max-episodes", type=int, default=50, help="Max episodes to process")
    parser.add_argument("--fps", type=int, default=10, help="Target FPS (default: 10)")
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    
    # Find all HDF5/MP4 pairs
    hdf5_files = sorted(input_dir.rglob("*.hdf5"))
    print(f"Found {len(hdf5_files)} HDF5 files in {input_dir}")
    
    episodes = []
    for i, hdf5_path in enumerate(tqdm(hdf5_files[:args.max_episodes], desc="Processing")):
        mp4_path = hdf5_path.with_suffix(".mp4")
        if not mp4_path.exists():
            print(f"  Skipping {hdf5_path.name}: no matching MP4")
            continue
        
        try:
            episode = process_episode(str(hdf5_path), str(mp4_path), episode_idx=i, target_fps=args.fps)
            episodes.append(episode)
            print(f"  Episode {i}: {episode['n_frames']} frames, task: {episode['task_description'][:60]}...")
        except Exception as e:
            print(f"  Error processing {hdf5_path.name}: {e}")
            continue
    
    if episodes:
        save_lerobot_dataset(episodes, args.output)
    else:
        print("No episodes processed!")


if __name__ == "__main__":
    main()
