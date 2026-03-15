"""Run DexMV retargeting on Modal GPU.

Uses DexMV's ChainMatchingPositionKinematicsRetargeting
to convert MANO joint positions → Adroit hand joint angles.

Usage: python -m modal run modal_dexmv_retarget.py --session stack2
"""
import modal
import json
import sys
import numpy as np
from pathlib import Path

app = modal.App("dexmv-retarget")

dexmv_image = (
    modal.Image.debian_slim(python_version="3.10")  # Use 3.10
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "numpy==1.23.5",
        "scipy",
        "natsort",
        "nlopt",
        "transforms3d",
        "torch",
        "pillow",
        "absl-py",
        "lxml",
        "six",
        "pyyaml",
        "gym",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/yzqin/dexmv-sim.git /dexmv",
        "cd /dexmv && pip install -e . --no-deps 2>&1 | tail -5",
        # Patch __init__.py to skip mujoco_py environments
        "echo '' > /dexmv/hand_imitation/env/environments/__init__.py",
        "echo '' > /dexmv/hand_imitation/env/__init__.py",
        "echo '' > /dexmv/hand_imitation/__init__.py",
        "echo 'import os; ASSETS_ROOT = os.path.join(os.path.dirname(__file__), \"../models/assets\")' >> /dexmv/hand_imitation/env/utils/mjcf_utils.py",
    )
)


@app.function(image=dexmv_image, timeout=600)
def run_dexmv_retarget(joints_data: list, global_data: list, hand_side: str) -> dict:
    """Run DexMV retargeting on provided joint + global data."""
    import sys
    sys.path.insert(0, "/dexmv")
    
    import numpy as np
    import os
    
    # Check if DexMV installed
    try:
        # Import retargeting directly, bypassing the mujoco_py environments
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "retargeting_optimizer", 
            "/dexmv/hand_imitation/kinematics/retargeting_optimizer.py"
        )
        from hand_imitation.kinematics.retargeting_optimizer import ChainMatchingPositionKinematicsRetargeting
        print("DexMV retargeting imported successfully")
    except ImportError as e:
        return {"error": f"DexMV import failed: {e}"}
    
    os.chdir("/dexmv/examples")
    
    assets_root = "/dexmv/hand_imitation/env/models/assets"
    path = os.path.join(assets_root, "adroit/adroit_relocate.xml")
    link_names = ["palm", "thmiddle", "ffmiddle", "mfmiddle", "rfmiddle", "lfmiddle",
                  "thtip", "fftip", "mftip", "rftip", "lftip"][:6]
    
    solver = ChainMatchingPositionKinematicsRetargeting(
        path, link_names, has_joint_limits=True, has_global_pose_limits=False
    )
    target_joint_index = [0, 2, 6, 10, 14, 18, 4, 8, 12, 16, 20][:6]
    
    hand_joint_seq = np.array(joints_data, dtype=np.float32)  # (N, 21, 3)
    hand_frame_seq = np.array(global_data, dtype=np.float32)  # (N, 16, 4, 4)
    
    print(f"Running retargeting: {hand_joint_seq.shape[0]} frames")
    
    robot_joints = solver.retarget(
        hand_joint_seq[:, target_joint_index, :],
        hand_frame_seq,
        name=f"stack2_{hand_side}",
        verbose=True
    )
    
    robot_joints_arr = np.array(robot_joints) if not isinstance(robot_joints, np.ndarray) else robot_joints
    print(f"Result shape: {robot_joints_arr.shape}")
    return {
        "robot_joints": robot_joints_arr.tolist(),
        "n_frames": len(robot_joints_arr),
        "dof": int(robot_joints_arr.shape[1]) if robot_joints_arr.ndim > 1 else 0,
    }


@app.local_entrypoint()
def main(session: str = "stack2"):
    base = Path(__file__).parent
    dexmv_dir = base / "dexmv_input" / session
    
    if not dexmv_dir.exists():
        print(f"DexMV input not found: {dexmv_dir}")
        print("Run: python convert_to_dexmv_format.py first")
        return
    
    for hand_side in ["right", "left"]:
        hand_dir = dexmv_dir / hand_side
        if not hand_dir.exists():
            continue
        
        # Load all frames (natural sort by frame number)
        import os, re
        def nat_sort(s):
            return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]
        
        joint_files = sorted([f for f in os.listdir(hand_dir) if f.endswith(".npy") and "joints" in f], key=nat_sort)
        global_files = sorted([f for f in os.listdir(hand_dir) if f.endswith(".npy") and "global" in f], key=nat_sort)
        
        seq_len = min(len(joint_files), len(global_files))
        print(f"{hand_side}: {seq_len} frames")
        
        joints_data = [np.load(str(hand_dir / f)).tolist() for f in joint_files[:seq_len]]
        global_data = [np.load(str(hand_dir / f)).tolist() for f in global_files[:seq_len]]
        
        result = run_dexmv_retarget.remote(joints_data, global_data, hand_side)
        
        if "error" in result:
            print(f"ERROR: {result['error']}")
            continue
        
        out_file = base / "retargeted" / f"{session}_{hand_side}_dexmv.json"
        out_file.write_text(json.dumps(result))
        print(f"Saved: {out_file} ({result['n_frames']} frames, {result['dof']} dof)")
