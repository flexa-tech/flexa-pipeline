"""Trim calibrated trajectory data to the action window.

Detects the meaningful manipulation segment using grasping signal + wrist velocity,
then slices the calibrated JSON arrays to that window.

Inserted between calibration (stage 5) and simulation (stage 6) in the pipeline.
"""
import json
import numpy as np
from pathlib import Path


def detect_action_window(wrist_sim, grasping, fps=10):
    """Detect the action window using grasping signal + wrist velocity.

    Returns (start_frame, end_frame) for trimming.

    Strategy:
    1. Compute wrist velocity with rolling mean (1-second window)
    2. Find sustained high-velocity region (top 30% of non-zero velocity)
    3. Find grasping frames within that region
    4. Pad with margins: 30 frames before, 50 frames after
    5. Enforce 15-30 second duration (150-300 frames at 10fps)

    Fallback chain:
    - velocity + grasping -> late grasping (>60%) -> velocity alone
    """
    wrist_sim = np.array(wrist_sim, dtype=float)
    grasping = np.array(grasping, dtype=float)
    n = len(wrist_sim)

    if n == 0:
        return 0, 0

    # Wrist velocity: frame-to-frame displacement with 1-second rolling mean
    disps = np.linalg.norm(np.diff(wrist_sim, axis=0), axis=1)
    vel = np.zeros(n)
    vel[1:] = disps
    window = max(1, fps)  # 1-second rolling window
    kernel = np.ones(window) / window
    rolling_vel = np.convolve(vel, kernel, mode='same')

    # High-velocity region: top 30% of non-zero velocity
    nonzero_vel = rolling_vel[rolling_vel > 0]
    if len(nonzero_vel) > 0:
        vel_threshold = np.percentile(nonzero_vel, 70)
        high_vel = rolling_vel > vel_threshold
    else:
        high_vel = np.ones(n, dtype=bool)

    # Find grasping frames in high-velocity region
    g = grasping > 0
    action_frames = np.where(g & high_vel)[0]

    if len(action_frames) == 0:
        # Fallback 1: use last 40% of grasping frames (mimics sim's late-grasp filter)
        grasp_idx = np.where(g)[0]
        if len(grasp_idx) > 0:
            cutoff = grasp_idx[int(len(grasp_idx) * 0.6)]
            action_frames = grasp_idx[grasp_idx >= cutoff]

    if len(action_frames) == 0:
        # Fallback 2: velocity alone (no grasping signal)
        action_frames = np.where(high_vel)[0]

    if len(action_frames) == 0:
        # Fallback 3: use entire trajectory
        print("  TRIM WARNING: No action window detected, using full trajectory")
        return 0, n

    # Margins from OUT-01 spec: "first grasp - 30 frames to last grasp + 50 frames"
    margin_before = 30   # 3 seconds at 10fps
    margin_after = 50    # 5 seconds at 10fps

    start = max(0, int(action_frames[0]) - margin_before)
    end = min(n, int(action_frames[-1]) + margin_after)

    # Enforce 15-30 second duration (150-300 frames at 10fps)
    duration = end - start
    target_min = 150  # 15s
    target_max = 300  # 30s

    if duration < target_min:
        # Expand symmetrically
        deficit = target_min - duration
        expand_before = deficit // 2
        expand_after = deficit - expand_before
        start = max(0, start - expand_before)
        end = min(n, end + expand_after)
        # Re-check if still short (hit array boundary)
        if (end - start) < target_min:
            # Expand from whichever side has room
            remaining = target_min - (end - start)
            if start > 0:
                start = max(0, start - remaining)
            else:
                end = min(n, end + remaining)
    elif duration > target_max:
        # Tighten: reduce margins but keep at least 10 frames before and 20 after action
        excess = duration - target_max
        reduce_before = min(excess // 2, max(0, start - max(0, int(action_frames[0]) - 10)))
        reduce_after = min(excess - reduce_before, max(0, end - (int(action_frames[-1]) + 20)))
        start += reduce_before
        end -= reduce_after

        # If still too long, action_frames span exceeds target_max.
        # Focus on the densest cluster: find the best target_max-sized sliding window
        # over action_frames that contains the most frames.
        if (end - start) > target_max:
            best_count = 0
            best_start_idx = 0
            for i in range(len(action_frames)):
                # Find how many action_frames fit in [action_frames[i], action_frames[i] + target_max - margin_before - margin_after)
                window_end = action_frames[i] + target_max - margin_before - margin_after
                count = np.searchsorted(action_frames, window_end, side='right') - i
                if count > best_count:
                    best_count = count
                    best_start_idx = i
            # Use the densest window
            focus_start = int(action_frames[best_start_idx])
            # Find the last action frame within this window
            window_end_frame = focus_start + target_max - margin_before - margin_after
            focus_end_idx = np.searchsorted(action_frames, window_end_frame, side='right') - 1
            focus_end = int(action_frames[focus_end_idx])
            start = max(0, focus_start - margin_before)
            end = min(n, focus_end + margin_after)

    return int(start), int(end)


def trim_calibrated_data(calib_path, start=None, end=None, fps=10):
    """Trim calibrated JSON arrays to the action window.

    If start/end not provided, auto-detects using detect_action_window().
    Modifies wrist_sim and grasping arrays in-place.
    Adds trim_info for traceability.

    Args:
        calib_path: Path to calibrated JSON file
        start: Start frame index (auto-detected if None)
        end: End frame index (auto-detected if None)
        fps: Frames per second (for duration calculation)

    Returns:
        dict with trim_info: {original_frames, start_frame, end_frame, trimmed_frames, duration_s}
    """
    calib_path = Path(calib_path)
    with open(calib_path) as f:
        calib = json.load(f)

    wrist_sim = calib["wrist_sim"]
    grasping = calib["grasping"]
    original_frames = len(wrist_sim)

    # Auto-detect window if not specified
    if start is None or end is None:
        start, end = detect_action_window(wrist_sim, grasping, fps=fps)

    # Validate bounds
    start = max(0, min(start, original_frames))
    end = max(start, min(end, original_frames))
    trimmed_frames = end - start

    if trimmed_frames == 0:
        print("  TRIM WARNING: Zero frames after trim, keeping full trajectory")
        return {"original_frames": original_frames, "start_frame": 0,
                "end_frame": original_frames, "trimmed_frames": original_frames,
                "duration_s": original_frames / fps}

    # Slice arrays
    calib["wrist_sim"] = wrist_sim[start:end]
    calib["grasping"] = grasping[start:end]

    # Add traceability metadata
    trim_info = {
        "original_frames": original_frames,
        "start_frame": start,
        "end_frame": end,
        "trimmed_frames": trimmed_frames,
        "duration_s": round(trimmed_frames / fps, 1),
    }
    calib["trim_info"] = trim_info

    # Write back
    with open(calib_path, "w") as f:
        json.dump(calib, f)

    return trim_info


def trim_and_save(session_name, calib_dir=None, start=None, end=None, fps=10):
    """High-level trim function for pipeline integration.

    Args:
        session_name: Session name (e.g., 'stack2')
        calib_dir: Directory containing calibrated JSONs (default: wrist_trajectories/)
        start: Manual start frame (auto-detect if None)
        end: Manual end frame (auto-detect if None)
        fps: Frames per second

    Returns:
        trim_info dict, or None on failure
    """
    if calib_dir is None:
        from pipeline_config import CALIB_DIR
        calib_dir = CALIB_DIR

    calib_path = Path(calib_dir) / f"{session_name}_calibrated.json"
    if not calib_path.exists():
        print(f"  ERROR: Calibrated file not found: {calib_path}")
        return None

    print(f"  Trimming: {calib_path}")
    trim_info = trim_calibrated_data(calib_path, start=start, end=end, fps=fps)

    print(f"  Original: {trim_info['original_frames']} frames ({trim_info['original_frames']/fps:.1f}s)")
    print(f"  Trimmed:  {trim_info['trimmed_frames']} frames ({trim_info['duration_s']}s)")
    print(f"  Window:   [{trim_info['start_frame']}, {trim_info['end_frame']})")

    return trim_info


if __name__ == "__main__":
    import sys
    session = sys.argv[1] if len(sys.argv) > 1 else "stack2"

    # Support optional manual start/end
    start = int(sys.argv[2]) if len(sys.argv) > 2 else None
    end = int(sys.argv[3]) if len(sys.argv) > 3 else None

    result = trim_and_save(session, start=start, end=end)
    if result:
        print(f"\n  Trim complete: {result['trimmed_frames']} frames ({result['duration_s']}s)")
    else:
        print("\n  Trim failed")
        sys.exit(1)
