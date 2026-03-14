"""Centralized path resolution for the flexa pipeline.

All scripts import paths from here instead of hardcoding.
"""
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MENAGERIE_ROOT = PROJECT_ROOT.parent / "mujoco_menagerie"

# Robot model directories
FRANKA_DIR = MENAGERIE_ROOT / "franka_emika_panda"
G1_DIR = MENAGERIE_ROOT / "unitree_g1"
H1_DIR = MENAGERIE_ROOT / "unitree_h1"
SHADOW_DIR = MENAGERIE_ROOT / "shadow_hand"

# Robot model paths (for Pinocchio / validation)
G1_URDF = PROJECT_ROOT / "models" / "unitree_g1" / "g1.urdf"

# Pipeline data directories
CALIB_DIR = PROJECT_ROOT / "wrist_trajectories"
OUT_DIR = PROJECT_ROOT / "sim_renders"
RAW_CAPTURES = PROJECT_ROOT / "raw_captures"
OBJECT_DET_DIR = PROJECT_ROOT / "object_detections"
RETARGET_DIR = PROJECT_ROOT / "gpu_retargeted"
R3D_OUTPUT = PROJECT_ROOT / "r3d_output"

# Ensure output directories exist
OUT_DIR.mkdir(exist_ok=True)
CALIB_DIR.mkdir(exist_ok=True)

# Tool paths
FFMPEG = shutil.which("ffmpeg")
