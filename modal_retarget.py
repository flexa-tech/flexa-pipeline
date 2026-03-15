"""Run dex-retargeting on Modal (Linux GPU) for HaMeR MANO → Shadow Hand.

Usage: python modal_retarget.py stack2
"""
import modal
import json
import sys
from pathlib import Path

app = modal.App("dex-retarget-shadow")

retarget_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install("numpy")
    .run_commands("pip uninstall pinocchio -y || true")  # remove wrong pinocchio package
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("dex_retargeting")
    .run_commands(
        "cd /root && git clone --depth 1 https://github.com/dexsuite/dex-urdf.git",
        "cp -r /root/dex-urdf/robots/hands/shadow_hand /root/shadow_hand",
        "ls /root/shadow_hand/",
    )
    .run_commands("python -c 'import dex_retargeting; print(\"OK\")'")
)

@app.function(image=retarget_image, timeout=300)
def retarget_session(hamer_data: dict, session: str) -> dict:
    """Retarget MANO joints to Shadow Hand joint angles."""
    import numpy as np
    from dex_retargeting.retargeting_config import RetargetingConfig
    from dex_retargeting.seq_retarget import SeqRetargeting
    import dex_retargeting
    from pathlib import Path as P

    pkg = P(dex_retargeting.__file__).parent

    # Find Shadow Hand URDF — search multiple locations
    import subprocess, glob
    
    # Try to find in dex_retargeting package
    urdf_path = None
    search_patterns = [
        str(pkg / "**" / "shadow_hand_right.urdf"),
        "/usr/local/lib/python3.11/site-packages/**/shadow_hand_right.urdf",
    ]
    for pattern in search_patterns:
        found = glob.glob(pattern, recursive=True)
        if found:
            urdf_path = found[0]
            break
    
    # If not found, try dex-urdf
    if not urdf_path:
        try:
            import dex_urdf
            urdf_pkg = P(dex_urdf.__file__).parent
            found = list(urdf_pkg.rglob("shadow_hand_right.urdf"))
            if found:
                urdf_path = str(found[0])
        except ImportError:
            pass
    
    if not urdf_path:
        # Use the appropriate hand URDF based on session name
        if "_left" in session:
            urdf_path = "/root/shadow_hand/shadow_hand_left.urdf"
        else:
            urdf_path = "/root/shadow_hand/shadow_hand_right.urdf"
    
    if not P(urdf_path).exists():
        return {"error": f"Shadow Hand URDF not found at {urdf_path}"}
    
    print(f"URDF: {urdf_path}")
    # Use VECTOR retargeting (palm→fingertip directions) — more robust than position
    config = RetargetingConfig(
        type="vector",
        urdf_path=urdf_path,
        target_origin_link_names=["palm"] * 10,
        target_task_link_names=[
            "thtip", "fftip", "mftip", "rftip", "lftip",
            "thmiddle", "ffmiddle", "mfmiddle", "rfmiddle", "lfmiddle",
        ],
        target_link_human_indices=np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # origins (wrist)
            [4, 8, 12, 16, 20, 2, 6, 10, 14, 18],  # targets (tips + middles)
        ]),
        scaling_factor=1.2,
        add_dummy_free_joint=True,
    )
    retargeting = config.build()
    joint_names = retargeting.joint_names
    print(f"Joint names ({len(joint_names)}): {joint_names}")

    results = []
    valid = 0
    total = len(hamer_data["results"])

    for r in hamer_data["results"]:
        frame_idx = r["frame_idx"]
        hands = r.get("hands", [])

        if not hands:
            results.append({"frame": frame_idx, "joints": None, "grasping": False})
            continue

        h = hands[0]
        j21 = h.get("joints_21", [])
        grasping = h.get("grasping", False)

        # Use keypoints_3d (HaMeR MANO 21x3) if available, else landmarks_3d
        kp3d = h.get("keypoints_3d", [])
        lm3d = h.get("landmarks_3d", [])
        
        if kp3d and len(kp3d) == 21 and isinstance(kp3d[0], list):
            # HaMeR MANO keypoints — 21x3, camera-relative, small scale (~0.1)
            joints_np = np.array(kp3d, dtype=float)
            # Scale to realistic hand size (~0.20m from wrist to middle fingertip)
            wrist = joints_np[0].copy()
            joints_np -= wrist  # center at origin
            current_scale = np.max(np.abs(joints_np))
            if current_scale > 0:
                joints_np *= (0.20 / current_scale)  # scale to ~20cm hand
        elif lm3d and len(lm3d) == 21:
            # MediaPipe landmarks — convert from dict format
            joints_np = np.array([[lm["x"], lm["y"], lm.get("z", 0)] for lm in lm3d], dtype=float)
            # Normalize pixel-space to hand-scale
            wrist = joints_np[0].copy()
            joints_np -= wrist
            scale = np.max(np.abs(joints_np[:, :2])) / 0.20
            if scale > 0:
                joints_np[:, :2] /= scale
                joints_np[:, 2] /= max(scale * 0.01, 0.01)
            joints_np += wrist * 0  # keep centered at origin
        else:
            results.append({"frame": frame_idx, "joints": None, "grasping": grasping})
            continue
        
        if joints_np.shape != (21, 3):
            results.append({"frame": frame_idx, "joints": None, "grasping": grasping})
            continue

        try:
            # Compute ref_value based on retargeting type
            indices = retargeting.optimizer.target_link_human_indices
            retargeting_type = retargeting.optimizer.retargeting_type
            if retargeting_type == "POSITION":
                ref_value = joints_np[indices, :]
            else:  # VECTOR
                origin_indices = indices[0, :]
                task_indices = indices[1, :]
                ref_value = joints_np[task_indices, :] - joints_np[origin_indices, :]
            
            robot_qpos = retargeting.retarget(ref_value)
            results.append({
                "frame": frame_idx,
                "joints": robot_qpos.tolist(),
                "grasping": grasping,
            })
            valid += 1
            # Debug: log first few frames
            if valid <= 3 or valid == 100:
                print(f"  F{frame_idx}: joints[6:12]={robot_qpos[6:12].round(3)} ref_range={joints_np.max()-joints_np.min():.4f}")
        except Exception as e:
            results.append({"frame": frame_idx, "joints": None, "grasping": grasping, "error": str(e)})

    print(f"Retargeted: {valid}/{total} frames")

    return {
        "session": session,
        "total_frames": total,
        "valid_frames": valid,
        "joint_names": joint_names,
        "results": results,
    }


@app.local_entrypoint()
def main(session: str = "stack2"):
    
    modal_results = Path(__file__).parent / "modal_results"
    
    # Prefer bimanual HaMeR output
    hamer_file = modal_results / f"{session}_hamer_bimanual.json"
    if not hamer_file.exists():
        hamer_file = modal_results / f"{session}_gpu_hands.json"
    
    if not hamer_file.exists():
        print(f"HaMeR data not found")
        return
    
    print(f"Loading {hamer_file.name}...")
    hamer_data = json.loads(hamer_file.read_text())
    print(f"  {len(hamer_data['results'])} frames, bimanual={hamer_data.get('bimanual_frames', '?')}")
    
    # Process left and right hands separately
    for hand_side in ["right", "left"]:
        print(f"\nRetargeting {hand_side} hand...")
        # Filter to only this hand's data
        filtered = {
            "results": [],
            "total_frames": hamer_data["total_frames"],
        }
        for r in hamer_data["results"]:
            hand_data = [h for h in r.get("hands", []) if h.get("hand") == hand_side]
            if hand_data:
                filtered["results"].append({
                    "frame_idx": r["frame_idx"],
                    "hands": hand_data[:1],  # take first match
                })
            else:
                filtered["results"].append({
                    "frame_idx": r["frame_idx"],
                    "hands": [],
                })
        
        result = retarget_session.remote(filtered, f"{session}_{hand_side}")
        
        if "error" in result:
            print(f"  ERROR: {result['error']}")
            continue
        
        out_dir = Path(__file__).parent / "retargeted"
        out_dir.mkdir(exist_ok=True)
        out_file = out_dir / f"{session}_{hand_side}_shadow_hand.json"
        out_file.write_text(json.dumps(result, indent=2))
        
        print(f"  {hand_side}: {result['valid_frames']}/{result['total_frames']} frames")
        print(f"  Saved: {out_file}")
