#!/usr/bin/env python3
"""Flexa Pipeline: End-to-end R3D -> robot simulation.

Processes iPhone LiDAR recordings (.r3d) through all stages and produces
a robot simulation video. Supports synthetic data mode for testing.

Usage:
    # Real R3D mode (primary):
    python run_pipeline.py --r3d path/to/recording.r3d --robot g1 --task stack

    # With manual object positions (skip GPU detection):
    python run_pipeline.py --r3d recording.r3d --robot g1 --task stack \
        --objects '[[0.5, 0.0, 0.43], [0.35, 0.1, 0.43]]'

    # Synthetic mode (testing without .r3d):
    python run_pipeline.py --synthetic --robot g1 --task stack
"""
import argparse
import json
import sys
import time
from pathlib import Path

from pipeline_config import PROJECT_ROOT, OUT_DIR, CALIB_DIR, R3D_OUTPUT, OBJECT_DET_DIR


def log_stage(stage_num, total, name, status="starting"):
    prefix = f"[{stage_num}/{total}]"
    if status == "starting":
        print(f"\n{'='*60}")
        print(f"{prefix} {name}")
        print(f"{'='*60}")
    elif status == "done":
        print(f"{prefix} {name} -- done")
    elif status == "skip":
        print(f"{prefix} {name} -- skipped")
    elif status == "fail":
        print(f"{prefix} {name} -- FAILED")


def validate_file(path, label):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{label}: {p} does not exist")
    if p.stat().st_size == 0:
        raise ValueError(f"{label}: {p} is empty")
    return True


def run_synthetic(robot, task, session_name=None):
    """Synthetic mode: generate test data and run simulation."""
    from synthetic_data import generate_synthetic

    total = 2
    session_name = session_name or f"synthetic_{task}"

    # Stage 1: Generate synthetic calibrated JSON
    log_stage(1, total, "Generate synthetic calibrated data")
    session, calib_path = generate_synthetic(robot, task)
    validate_file(calib_path, "Calibrated JSON")
    log_stage(1, total, "Generate synthetic calibrated data", "done")

    # Stage 2: Run simulation
    log_stage(2, total, f"Run {robot} simulation")
    video_path = run_simulation(robot, session, task)
    log_stage(2, total, f"Run {robot} simulation", "done")

    return video_path


def run_r3d_pipeline(r3d_path, robot, task, session_name=None, objects_manual=None,
                     trim_enabled=False, trim_start=None, trim_end=None,
                     use_hamer=True):
    """Real R3D mode: full pipeline from .r3d to simulation video."""
    total = 7
    r3d_path = Path(r3d_path).resolve()

    if not r3d_path.exists():
        print(f"ERROR: R3D file not found: {r3d_path}", file=sys.stderr)
        sys.exit(1)

    if session_name is None:
        session_name = r3d_path.stem.replace(" ", "_").lower()

    session_dir = R3D_OUTPUT / session_name
    R3D_OUTPUT.mkdir(parents=True, exist_ok=True)

    # Stage 1: Ingest R3D
    log_stage(1, total, "Ingest R3D -> frames + depth + video")
    from r3d_ingest import ingest_r3d
    ingest_result = ingest_r3d(
        str(r3d_path), str(session_dir),
        task_category=task,
        task_description=f"{task} manipulation task",
    )

    # Validate
    video_path = ingest_result.get("video_path", "")
    if not video_path or not Path(video_path).exists():
        video_path = str(session_dir / "video.mp4")
    validate_file(video_path, "Extracted video")
    n_frames = ingest_result.get("num_frames", 0)
    print(f"  Extracted {n_frames} frames, video: {video_path}")
    log_stage(1, total, "Ingest R3D", "done")

    # Stage 2: Hand tracking
    # Try HaMeR (GPU, via Modal) first, fall back to MediaPipe
    log_stage(2, total, "Hand tracking -> trajectory JSON")
    traj_path = str(session_dir / f"{session_name}_hand_trajectory.json")
    trajectory = _run_hand_tracking(video_path, session_name, session_dir, traj_path,
                                     use_hamer=use_hamer)
    log_stage(2, total, "Hand tracking", "done")

    # Prepare retarget data + copy R3D (prerequisites for object detection and wrist reconstruction)
    _build_retarget_data(session_name, session_dir, trajectory)

    from pipeline_config import RAW_CAPTURES
    raw_session_dir = RAW_CAPTURES / session_name
    raw_session_dir.mkdir(parents=True, exist_ok=True)
    r3d_link = raw_session_dir / r3d_path.name
    if not r3d_link.exists():
        import shutil
        shutil.copy2(str(r3d_path), str(r3d_link))

    # Stage 3: Object detection
    log_stage(3, total, "Object detection -> objects JSON")
    OBJECT_DET_DIR.mkdir(parents=True, exist_ok=True)

    if objects_manual is not None:
        obj_positions = json.loads(objects_manual) if isinstance(objects_manual, str) else objects_manual
        labels = [f"block_{chr(97+i)}" for i in range(len(obj_positions))]
        _write_object_detections(session_name, labels, obj_positions)
        print(f"  Manual object positions: {len(obj_positions)} objects")
    else:
        # Use GroundingDINO for detection (requires GPU or transformers)
        _run_object_detection(session_name, r3d_path, trajectory)
    log_stage(3, total, "Object detection", "done")

    # Stage 4: 3D wrist reconstruction
    log_stage(4, total, "3D wrist reconstruction -> wrist3d JSON")

    from reconstruct_wrist_3d import process_session
    wrist_result = process_session(session_name)

    if wrist_result:
        n_valid = wrist_result.get("n_valid", 0)
        n_total = wrist_result.get("n_frames", 1)
        valid_rate = n_valid / max(n_total, 1)
        print(f"  Valid frames: {valid_rate:.0%} ({n_valid}/{n_total})")
        if valid_rate < 0.3:
            print(f"  WARNING: Low valid frame rate ({valid_rate:.0%})")
    else:
        print("  WARNING: Wrist reconstruction returned no result")
    log_stage(4, total, "3D wrist reconstruction", "done")

    # Stage 5: Workspace calibration
    log_stage(5, total, "Workspace calibration -> calibrated JSON")
    from calibrate_workspace import calibrate_session
    calib_path = calibrate_session(session_name)

    if calib_path is None:
        print("  ERROR: Calibration failed")
        sys.exit(1)
    validate_file(calib_path, "Calibrated JSON")
    log_stage(5, total, "Workspace calibration", "done")

    # Stage 6: Trim trajectory (optional, enabled by --trim)
    if trim_enabled:
        log_stage(6, total, "Trim trajectory -> action window")
        from trim_trajectory import trim_and_save
        trim_info = trim_and_save(
            session_name,
            start=trim_start,
            end=trim_end,
        )
        if trim_info:
            duration = trim_info['duration_s']
            print(f"  Trimmed to {trim_info['trimmed_frames']} frames ({duration}s)")
        else:
            print("  WARNING: Trim failed, using full trajectory")
        log_stage(6, total, "Trim trajectory", "done")
    else:
        log_stage(6, total, "Trim trajectory", "skip")

    # Stage 7: Simulation
    log_stage(7, total, f"Run {robot} simulation -> video")
    video_path = run_simulation(robot, session_name, task)
    log_stage(7, total, f"Run {robot} simulation", "done")

    return video_path


def _run_hand_tracking(video_path, session_name, session_dir, traj_path,
                       use_hamer=True):
    """Run hand tracking: try HaMeR via Modal, fall back to MediaPipe."""
    trajectory = None

    # Try HaMeR (GPU) first
    if use_hamer:
        try:
            from egocrowd.hand_pose import extract_hand_poses
            frames_dir = str(session_dir / "frames")
            result = extract_hand_poses(frames_dir)
            # Convert HaMeR output to trajectory format
            if result:
                trajectory = _hamer_to_trajectory(result, video_path)
                print("  Hand tracking: HaMeR (GPU)")
        except (ImportError, NotImplementedError, Exception) as e:
            print(f"  HaMeR unavailable ({type(e).__name__}), using MediaPipe")
    else:
        print("  HaMeR disabled (--no-hamer), using MediaPipe")

    # Fall back to MediaPipe
    if trajectory is None:
        from hand_tracker_v2 import process_video
        trajectory = process_video(video_path, traj_path)
        print("  Hand tracking: MediaPipe (CPU)")

    if trajectory and trajectory.get("frames"):
        frames_with_hands = sum(1 for f in trajectory["frames"] if f.get("hands"))
        total_frames = len(trajectory["frames"])
        detection_rate = frames_with_hands / max(total_frames, 1)
        print(f"  Detection rate: {detection_rate:.0%} ({frames_with_hands}/{total_frames})")
        if detection_rate < 0.1:
            print(f"  WARNING: Very low hand detection rate ({detection_rate:.0%})")

    return trajectory


def _hamer_to_trajectory(hamer_result, video_path):
    """Convert HaMeR output to the trajectory dict format used downstream."""
    # HaMeR returns per-frame MANO params with wrist positions and grasping
    # Map to the same format as hand_tracker_v2 output
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    trajectory = {
        "metadata": {
            "source": "hamer",
            "video": str(Path(video_path).name),
            "fps": fps,
            "resolution": [w, h],
            "total_frames": total,
        },
        "frames": [],
    }

    timesteps = hamer_result.get("timesteps", [])
    for i, ts in enumerate(timesteps):
        frame_data = {
            "frame": i,
            "timestamp": round(i / fps, 4),
            "hands": [],
        }
        if ts.get("wrist_pixel") or ts.get("detected"):
            wp = ts.get("wrist_pixel", [0, 0])
            hand_data = {
                "hand": "right",
                "confidence": ts.get("confidence", 0.9),
                "wrist": {
                    "position": {"x": wp[0], "y": wp[1], "z": 0},
                },
                "grasping": ts.get("grasping", False),
            }
            # TRK-05: Pass through HaMeR 3D wrist position (camera frame)
            if ts.get("wrist_3d_camera"):
                hand_data["wrist_3d_camera"] = ts["wrist_3d_camera"]
            if ts.get("joints_3d"):
                hand_data["joints_3d"] = ts["joints_3d"]
            frame_data["hands"].append(hand_data)
        trajectory["frames"].append(frame_data)

    return trajectory


def _run_object_detection(session_name, r3d_path, trajectory):
    """Run GroundingDINO object detection. Errors out if unavailable."""
    try:
        from detect_objects import process_session
        result = process_session(session_name)
        if result and result.get("detections"):
            # Convert GroundingDINO output to the format calibrate_workspace expects
            detections = result["detections"]
            labels = [d.get("label", f"object_{i}") for i, d in enumerate(detections)]
            positions = [d["pos_world"] for d in detections if d.get("pos_world")]
            if positions:
                _write_object_detections(session_name, labels[:len(positions)], positions)
                print(f"  GroundingDINO detected {len(positions)} objects with 3D positions")
                return
            else:
                print("  WARNING: GroundingDINO detected objects but no 3D positions (depth missing?)")
        else:
            print("  WARNING: GroundingDINO returned no detections")
    except ImportError as e:
        print(f"  GroundingDINO unavailable: {e}")
    except Exception as e:
        print(f"  GroundingDINO failed: {e}")

    print("  ERROR: Object detection failed. Provide positions manually with --objects", file=sys.stderr)
    print("  Example: --objects '[[0.5, 0.0, 0.43], [0.35, 0.1, 0.43]]'", file=sys.stderr)
    sys.exit(1)


def _write_object_detections(session_name, labels, positions):
    """Write object detections to JSON in the format downstream stages expect."""
    OBJECT_DET_DIR.mkdir(parents=True, exist_ok=True)
    detections = []
    for label, pos in zip(labels, positions):
        detections.append({
            "label": label,
            "pos_world": pos,
            "confidence": 1.0,
            "source": "manual",
        })
    result = {"detections": detections}
    out_path = OBJECT_DET_DIR / f"{session_name}_objects_clean.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved {len(result['detections'])} detections to {out_path}")


def _build_retarget_data(session_name, session_dir, trajectory):
    """Convert hand_tracker_v2 output to retarget-compatible format for reconstruct_wrist_3d."""
    from pipeline_config import RETARGET_DIR
    RETARGET_DIR.mkdir(parents=True, exist_ok=True)

    timesteps = []
    if trajectory and trajectory.get("frames"):
        for frame in trajectory["frames"]:
            ts = {"frame": frame["frame"], "timestamp": frame.get("timestamp", 0)}
            if frame.get("hands"):
                hand = frame["hands"][0]  # primary hand
                wrist_pos = hand.get("wrist", {}).get("position", {})
                ts["wrist_pixel"] = [wrist_pos.get("x", 0), wrist_pos.get("y", 0)]
                ts["grasping"] = hand.get("grasping", False)
                # TRK-05: Pass through HaMeR camera-frame 3D wrist
                if hand.get("wrist_3d_camera"):
                    ts["wrist_3d_camera"] = hand["wrist_3d_camera"]
            else:
                ts["wrist_pixel"] = None
                ts["grasping"] = False
            timesteps.append(ts)

    retarget_data = {"session": session_name, "timesteps": timesteps}
    out_path = RETARGET_DIR / f"{session_name}_retargeted.json"
    with open(out_path, "w") as f:
        json.dump(retarget_data, f)
    print(f"  Built retarget data: {out_path} ({len(timesteps)} timesteps)")


def run_simulation(robot, session_name, task):
    """Run the appropriate simulation script for the given robot."""
    if robot == "g1":
        from mujoco_g1_v10 import render_task
        video_path = render_task(session_name)
    elif robot == "franka":
        from mujoco_franka_v9 import simulate
        simulate(session_name, task)
        video_path = OUT_DIR / f"{session_name}_franka_v9.mp4"
    elif robot == "h1":
        from mujoco_h1_shadow_v1 import render_task
        video_path = render_task(session_name)
    else:
        raise ValueError(f"Unknown robot: {robot}")

    video_path = Path(video_path) if video_path else OUT_DIR / f"{session_name}_{robot}.mp4"
    if video_path.exists() and video_path.stat().st_size > 0:
        size_kb = video_path.stat().st_size // 1024
        print(f"\n  Output video: {video_path} ({size_kb} KB)")
    else:
        print(f"\n  WARNING: Output video missing or empty: {video_path}")

    return str(video_path)


def main():
    parser = argparse.ArgumentParser(
        description="Flexa Pipeline: R3D -> robot simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --synthetic --robot g1 --task stack
  python run_pipeline.py --r3d recording.r3d --robot g1 --task stack
  python run_pipeline.py --r3d recording.r3d --robot g1 --task stack \\
      --objects '[[0.5, 0.0, 0.43], [0.35, 0.1, 0.43]]'
""")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--r3d", type=str, help="Path to .r3d file (real mode)")
    group.add_argument("--synthetic", action="store_true", help="Use synthetic data (test mode)")

    parser.add_argument("--robot", default="g1", choices=["g1", "franka", "h1"],
                        help="Robot model (default: g1)")
    parser.add_argument("--task", default="stack", choices=["stack", "pick_place", "sort"],
                        help="Task type (default: stack)")
    parser.add_argument("--objects", type=str, default=None,
                        help="Manual object positions as JSON [[x,y,z], ...] (skip detection)")
    parser.add_argument("--session", type=str, default=None,
                        help="Session name (default: derived from filename)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: sim_renders/)")
    parser.add_argument("--hamer", action="store_true", default=True,
                        help="Try HaMeR for hand tracking (default: True)")
    parser.add_argument("--no-hamer", dest="hamer", action="store_false",
                        help="Skip HaMeR, use MediaPipe only")
    parser.add_argument("--trim", action="store_true", default=False,
                        help="Trim trajectory to action window (15-30s)")
    parser.add_argument("--trim-start", type=int, default=None,
                        help="Manual trim start frame (overrides auto-detection)")
    parser.add_argument("--trim-end", type=int, default=None,
                        help="Manual trim end frame (overrides auto-detection)")
    args = parser.parse_args()

    if args.output:
        import pipeline_config
        pipeline_config.OUT_DIR = Path(args.output)
        pipeline_config.OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Flexa Pipeline -- robot={args.robot}, task={args.task}")
    print(f"{'='*60}")
    t0 = time.time()

    if args.synthetic:
        video = run_synthetic(args.robot, args.task, args.session)
    else:
        video = run_r3d_pipeline(
            args.r3d, args.robot, args.task, args.session, args.objects,
            trim_enabled=args.trim,
            trim_start=args.trim_start,
            trim_end=args.trim_end,
            use_hamer=args.hamer,
        )

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Pipeline complete in {elapsed:.1f}s")
    print(f"Output: {video}")


if __name__ == "__main__":
    main()
