"""
Flexa → Open X-Embodiment (OXE) Format Converter
Converts Apple Flexa egocentric manipulation data to RLDS-compliant TFRecord format
compatible with the Open X-Embodiment dataset ecosystem.

OXE uses the RLDS episode format (google-research/rlds) stored as TFRecords via
tensorflow_datasets. Each dataset is a sequence of episodes, each episode a sequence
of steps with: observation, action, reward, discount, is_first, is_last, is_terminal,
language_instruction, and language_embedding.

Reference: https://github.com/kpertsch/rlds_dataset_builder
Paper: https://arxiv.org/abs/2310.08864

Usage:
    # Step 1: Build the dataset
    cd pipeline
    python Flexa_to_oxe.py --input /data/Flexa/test --max-episodes 50

    # This creates ~/tensorflow_datasets/Flexa_dataset/1.0.0/

    # Step 2: Load with standard RLDS tooling
    import tensorflow_datasets as tfds
    ds = tfds.load('Flexa_dataset', split='train')

Requirements:
    pip install tensorflow tensorflow_datasets tensorflow_hub numpy h5py Pillow
"""

from typing import Iterator, Tuple, Any

import argparse
import glob
import os
import subprocess
import tempfile
from pathlib import Path

import h5py
import numpy as np

# Lazy imports for TF (heavy)
tf = None
tfds = None
hub = None


def _ensure_tf():
    global tf, tfds, hub
    if tf is None:
        import tensorflow as _tf
        import tensorflow_datasets as _tfds
        import tensorflow_hub as _hub
        tf = _tf
        tfds = _tfds
        hub = _hub


# Key hand joints — same as Flexa_to_lerobot.py
KEY_JOINTS = [
    "leftHand", "rightHand",
    "leftIndexFingerTip", "leftIndexFingerKnuckle",
    "leftMiddleFingerTip", "leftMiddleFingerKnuckle",
    "leftThumbTip", "leftThumbKnuckle",
    "rightIndexFingerTip", "rightIndexFingerKnuckle",
    "rightMiddleFingerTip", "rightMiddleFingerKnuckle",
    "rightThumbTip", "rightThumbKnuckle",
]

# State dimension: 14 joints × 9 (3 pos + 6 rot) = 126
STATE_DIM = len(KEY_JOINTS) * 9
# Action = delta state, same dim
ACTION_DIM = STATE_DIM
# Image size for OXE (standard is 256×256, resized to 128×128 for training transform)
IMAGE_SIZE = 256


def extract_joint_state(hdf5_file, frame_idx, joints=KEY_JOINTS):
    """Extract flattened joint positions from a single frame."""
    state = []
    for joint in joints:
        key = f"transforms/{joint}"
        if key in hdf5_file:
            transform = hdf5_file[key][frame_idx]  # 4x4 SE3
            position = transform[:3, 3]
            rotation = transform[:3, :3]
            rot_6d = rotation[:, :2].flatten()
            state.extend(position.tolist())
            state.extend(rot_6d.tolist())
        else:
            state.extend([0.0] * 9)
    return np.array(state, dtype=np.float32)


def extract_video_frames(mp4_path, target_fps=10, max_frames=None):
    """Extract frames from MP4 at target FPS using ffmpeg."""
    from PIL import Image as PILImage

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
        img = PILImage.open(f).resize((IMAGE_SIZE, IMAGE_SIZE))
        images.append(np.array(img, dtype=np.uint8))
        os.remove(f)
    os.rmdir(tmpdir)

    return images


def load_episode_raw(hdf5_path, mp4_path, target_fps=10):
    """Load a single Flexa episode and return (states, frames, description)."""
    with h5py.File(hdf5_path, "r") as f:
        n_frames = f["transforms/camera"].shape[0]
        source_fps = 30
        step = max(1, source_fps // target_fps)

        frame_indices = list(range(0, n_frames, step))
        states = [extract_joint_state(f, idx) for idx in frame_indices]

        description = ""
        if "llm_description" in f.attrs:
            description = f.attrs["llm_description"]

    frames = extract_video_frames(mp4_path, target_fps=target_fps, max_frames=len(states))

    min_len = min(len(states), len(frames))
    states = states[:min_len]
    frames = frames[:min_len]

    return np.array(states, dtype=np.float32), frames, description


# ---------------------------------------------------------------------------
# RLDS / TFDS Dataset Builder
# ---------------------------------------------------------------------------

def _get_builder_class(input_dir: str, max_episodes: int, fps: int):
    """Dynamically create the TFDS builder class with captured config."""
    _ensure_tf()

    class FlexaDataset(tfds.core.GeneratorBasedBuilder):
        """RLDS dataset builder for Flexa egocentric hand manipulation data."""

        VERSION = tfds.core.Version("1.0.0")
        RELEASE_NOTES = {
            "1.0.0": "Initial release — Flexa hand tracking converted to OXE/RLDS.",
        }

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._embed = hub.load(
                "https://tfhub.dev/google/universal-sentence-encoder-large/5"
            )
            self._input_dir = input_dir
            self._max_episodes = max_episodes
            self._fps = fps

        def _info(self) -> tfds.core.DatasetInfo:
            return self.dataset_info_from_configs(
                features=tfds.features.FeaturesDict({
                    "steps": tfds.features.Dataset({
                        "observation": tfds.features.FeaturesDict({
                            "image": tfds.features.Image(
                                shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                                dtype=np.uint8,
                                encoding_format="jpeg",
                                doc="Egocentric RGB camera observation (Apple Vision Pro).",
                            ),
                            "state": tfds.features.Tensor(
                                shape=(STATE_DIM,),
                                dtype=np.float32,
                                doc=f"Hand joint state vector ({len(KEY_JOINTS)} joints × "
                                    f"9 [3 pos + 6D rot] = {STATE_DIM} dims).",
                            ),
                        }),
                        "action": tfds.features.Tensor(
                            shape=(ACTION_DIM,),
                            dtype=np.float32,
                            doc=f"Delta hand joint action ({ACTION_DIM} dims, "
                                f"same structure as state).",
                        ),
                        "discount": tfds.features.Scalar(
                            dtype=np.float32,
                            doc="Discount factor, default 1.0.",
                        ),
                        "reward": tfds.features.Scalar(
                            dtype=np.float32,
                            doc="Reward: 1.0 on final step (demonstrations), 0.0 otherwise.",
                        ),
                        "is_first": tfds.features.Scalar(
                            dtype=np.bool_,
                            doc="True on first step of the episode.",
                        ),
                        "is_last": tfds.features.Scalar(
                            dtype=np.bool_,
                            doc="True on last step of the episode.",
                        ),
                        "is_terminal": tfds.features.Scalar(
                            dtype=np.bool_,
                            doc="True on last step (terminal for demonstrations).",
                        ),
                        "language_instruction": tfds.features.Text(
                            doc="Natural language task description (from LLM annotation).",
                        ),
                        "language_embedding": tfds.features.Tensor(
                            shape=(512,),
                            dtype=np.float32,
                            doc="Universal Sentence Encoder embedding of language_instruction.",
                        ),
                    }),
                    "episode_metadata": tfds.features.FeaturesDict({
                        "file_path": tfds.features.Text(
                            doc="Path to the original Flexa HDF5 file.",
                        ),
                        "fps": tfds.features.Scalar(
                            dtype=np.int32,
                            doc="Frame rate of the episode.",
                        ),
                    }),
                }),
            )

        def _split_generators(self, dl_manager):
            return {
                "train": self._generate_examples(),
            }

        def _generate_examples(self) -> Iterator[Tuple[str, Any]]:
            hdf5_files = sorted(Path(self._input_dir).rglob("*.hdf5"))
            hdf5_files = hdf5_files[: self._max_episodes]

            for hdf5_path in hdf5_files:
                mp4_path = hdf5_path.with_suffix(".mp4")
                if not mp4_path.exists():
                    continue

                try:
                    states, frames, description = load_episode_raw(
                        str(hdf5_path), str(mp4_path), target_fps=self._fps
                    )
                except Exception as e:
                    print(f"Skipping {hdf5_path.name}: {e}")
                    continue

                n = len(states)
                if n < 2:
                    continue

                # Compute delta actions
                actions = np.diff(states, axis=0)
                actions = np.concatenate(
                    [actions, np.zeros((1, STATE_DIM), dtype=np.float32)], axis=0
                )

                # Language embedding
                lang = description if description else "manipulate objects with hands"
                lang_embedding = self._embed([lang])[0].numpy().astype(np.float32)

                # Build steps
                steps = []
                for i in range(n):
                    steps.append({
                        "observation": {
                            "image": frames[i],
                            "state": states[i],
                        },
                        "action": actions[i],
                        "discount": 1.0,
                        "reward": float(i == n - 1),
                        "is_first": i == 0,
                        "is_last": i == n - 1,
                        "is_terminal": i == n - 1,
                        "language_instruction": lang,
                        "language_embedding": lang_embedding,
                    })

                sample = {
                    "steps": steps,
                    "episode_metadata": {
                        "file_path": str(hdf5_path),
                        "fps": self._fps,
                    },
                }

                yield str(hdf5_path), sample

    return FlexaDataset


# ---------------------------------------------------------------------------
# Standalone converter (no tfds build CLI needed)
# ---------------------------------------------------------------------------

def convert_standalone(input_dir: str, output_dir: str, max_episodes: int, fps: int):
    """
    Convert Flexa data to OXE-compatible RLDS TFRecords without requiring
    the full tfds build pipeline. Writes TFRecords + metadata directly.

    This is useful when you can't install the full TFDS builder infrastructure
    or want a simpler, more portable output.
    """
    _ensure_tf()
    from PIL import Image as PILImage

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load sentence encoder for language embeddings
    print("Loading Universal Sentence Encoder...")
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    hdf5_files = sorted(Path(input_dir).rglob("*.hdf5"))[:max_episodes]
    print(f"Found {len(hdf5_files)} episodes to convert")

    writer = tf.io.TFRecordWriter(str(output_path / "Flexa_dataset-train.tfrecord"))
    episode_count = 0

    for hdf5_path in hdf5_files:
        mp4_path = hdf5_path.with_suffix(".mp4")
        if not mp4_path.exists():
            continue

        try:
            states, frames, description = load_episode_raw(
                str(hdf5_path), str(mp4_path), target_fps=fps
            )
        except Exception as e:
            print(f"  Skipping {hdf5_path.name}: {e}")
            continue

        n = len(states)
        if n < 2:
            continue

        actions = np.diff(states, axis=0)
        actions = np.concatenate(
            [actions, np.zeros((1, STATE_DIM), dtype=np.float32)], axis=0
        )

        lang = description if description else "manipulate objects with hands"
        lang_embedding = embed([lang])[0].numpy().astype(np.float32)

        # Serialize episode as a SequenceExample (RLDS convention)
        step_features = {
            "observation/image": [],
            "observation/state": [],
            "action": [],
            "discount": [],
            "reward": [],
            "is_first": [],
            "is_last": [],
            "is_terminal": [],
            "language_instruction": [],
            "language_embedding": [],
        }

        for i in range(n):
            # Encode image as JPEG bytes
            img = PILImage.fromarray(frames[i])
            import io
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            img_bytes = buf.getvalue()

            step_features["observation/image"].append(
                tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes]))
            )
            step_features["observation/state"].append(
                tf.train.Feature(float_list=tf.train.FloatList(value=states[i].tolist()))
            )
            step_features["action"].append(
                tf.train.Feature(float_list=tf.train.FloatList(value=actions[i].tolist()))
            )
            step_features["discount"].append(
                tf.train.Feature(float_list=tf.train.FloatList(value=[1.0]))
            )
            step_features["reward"].append(
                tf.train.Feature(float_list=tf.train.FloatList(value=[float(i == n - 1)]))
            )
            step_features["is_first"].append(
                tf.train.Feature(int64_list=tf.train.Int64List(value=[int(i == 0)]))
            )
            step_features["is_last"].append(
                tf.train.Feature(int64_list=tf.train.Int64List(value=[int(i == n - 1)]))
            )
            step_features["is_terminal"].append(
                tf.train.Feature(int64_list=tf.train.Int64List(value=[int(i == n - 1)]))
            )
            step_features["language_instruction"].append(
                tf.train.Feature(bytes_list=tf.train.BytesList(value=[lang.encode("utf-8")]))
            )
            step_features["language_embedding"].append(
                tf.train.Feature(float_list=tf.train.FloatList(value=lang_embedding.tolist()))
            )

        # Build SequenceExample
        feature_lists = {
            k: tf.train.FeatureList(feature=v) for k, v in step_features.items()
        }
        context = tf.train.Features(feature={
            "episode_metadata/file_path": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[str(hdf5_path).encode("utf-8")])
            ),
            "episode_metadata/fps": tf.train.Feature(
                int64_list=tf.train.Int64List(value=[fps])
            ),
        })

        seq_example = tf.train.SequenceExample(
            context=context,
            feature_lists=tf.train.FeatureLists(feature_list=feature_lists),
        )

        writer.write(seq_example.SerializeToString())
        episode_count += 1
        print(f"  Episode {episode_count}: {n} steps from {hdf5_path.name}")

    writer.close()

    # Write metadata JSON
    import json
    metadata = {
        "dataset_name": "Flexa_dataset",
        "format": "OXE/RLDS",
        "version": "1.0.0",
        "n_episodes": episode_count,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "image_size": IMAGE_SIZE,
        "joints": KEY_JOINTS,
        "description": (
            "Flexa egocentric hand manipulation data converted to "
            "Open X-Embodiment (RLDS) format. Hand pose from Apple Vision Pro "
            "with 14 key joints (3D position + 6D rotation per joint)."
        ),
    }
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nOXE dataset saved to {output_path}")
    print(f"  Episodes: {episode_count}")
    print(f"  Format: RLDS TFRecord (Open X-Embodiment compatible)")
    print(f"  State dim: {STATE_DIM}, Action dim: {ACTION_DIM}")

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Convert Flexa to Open X-Embodiment (OXE/RLDS) format"
    )
    parser.add_argument("--input", required=True, help="Path to Flexa data directory")
    parser.add_argument(
        "--output", default=None,
        help="Output directory (default: ~/tensorflow_datasets/Flexa_dataset)"
    )
    parser.add_argument("--max-episodes", type=int, default=50)
    parser.add_argument("--fps", type=int, default=10, help="Target FPS")
    parser.add_argument(
        "--mode", choices=["tfds", "standalone"], default="standalone",
        help="'tfds' uses tensorflow_datasets builder (requires tfds build), "
             "'standalone' writes TFRecords directly (simpler, portable)."
    )
    args = parser.parse_args()

    if args.mode == "tfds":
        print("TFDS builder mode: use 'tfds build' from the dataset directory.")
        print("Creating builder class for programmatic use...")
        BuilderClass = _get_builder_class(
            args.input, args.max_episodes, args.fps
        )
        builder = BuilderClass(data_dir=args.output or os.path.expanduser("~/tensorflow_datasets"))
        builder.download_and_prepare()
        print("Done! Dataset available via: tfds.load('Flexa_dataset')")
    else:
        output = args.output or os.path.expanduser("~/tensorflow_datasets/Flexa_dataset")
        convert_standalone(args.input, output, args.max_episodes, args.fps)


if __name__ == "__main__":
    main()
