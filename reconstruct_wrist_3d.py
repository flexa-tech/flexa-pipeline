"""Reconstruct 3D wrist trajectories from wrist_pixel + R3D depth maps.
For each frame with a wrist_pixel, look up depth, unproject to 3D, transform to world.
Then normalize all world trajectories to a common sim workspace.
"""
import json, sys
import numpy as np
from pathlib import Path
import zipfile, io
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.signal import savgol_filter

RAW = Path(__file__).resolve().parent / "raw_captures"
RETARGET = Path(__file__).resolve().parent / "gpu_retargeted"
OUTPUT = Path(__file__).resolve().parent / "wrist_trajectories"
OUTPUT.mkdir(exist_ok=True)


def load_r3d_depth(r3d_path, frame_idx):
    """Load depth map for a specific R3D frame."""
    import liblzfse
    with zipfile.ZipFile(r3d_path) as z:
        meta = json.load(z.open("metadata"))
        with z.open(f"rgbd/{frame_idx}.depth") as f:
            raw = liblzfse.decompress(f.read())
        depth = np.frombuffer(raw, dtype=np.float32).reshape(meta["dh"], meta["dw"])
    return meta, depth


def pixel_to_world(u, v, depth_map, meta, r3d_frame):
    """Unproject pixel to world coordinates."""
    dh, dw = meta["dh"], meta["dw"]
    h, w = meta["h"], meta["w"]

    # Intrinsics
    if meta.get("perFrameIntrinsicCoeffs") and r3d_frame < len(meta["perFrameIntrinsicCoeffs"]):
        c = meta["perFrameIntrinsicCoeffs"][r3d_frame]
        fx, fy, cx, cy = c[0], c[1], c[2], c[3]
    else:
        K = meta["K"]
        fx, fy, cx, cy = K[0], K[4], K[2], K[5]

    # Depth lookup (median of 5x5 patch)
    du = int(np.clip(u * dw / w, 0, dw-1))
    dv = int(np.clip(v * dh / h, 0, dh-1))
    r = 3
    patch = depth_map[max(0,dv-r):dv+r+1, max(0,du-r):du+r+1]
    valid = patch[(patch > 0.05) & (patch < 3.0)]
    if len(valid) == 0:
        return None
    Z = float(np.median(valid))

    # Unproject to camera coords
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    point_cam = np.array([X, Y, Z])

    # Camera → world
    pose_idx = min(r3d_frame, len(meta["poses"]) - 1)
    pose = meta["poses"][pose_idx]
    tx, ty, tz = pose[0], pose[1], pose[2]
    qw, qx, qy, qz = pose[3], pose[4], pose[5], pose[6]
    R = np.array([
        [1-2*(qy**2+qz**2), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
        [2*(qx*qy+qw*qz), 1-2*(qx**2+qz**2), 2*(qy*qz-qw*qx)],
        [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx**2+qy**2)]
    ])
    point_world = R @ point_cam + np.array([tx, ty, tz])
    return point_world


def cam_to_world(point_cam, pose):
    """Transform a camera-frame point to world frame using R3D pose.

    Args:
        point_cam: (3,) array-like, point in camera coordinates
        pose: R3D pose [tx, ty, tz, qw, qx, qy, qz]

    Returns:
        (3,) ndarray, point in world coordinates
    """
    tx, ty, tz = pose[0], pose[1], pose[2]
    qw, qx, qy, qz = pose[3], pose[4], pose[5], pose[6]
    R = np.array([
        [1-2*(qy**2+qz**2), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
        [2*(qx*qy+qw*qz), 1-2*(qx**2+qz**2), 2*(qy*qz-qw*qx)],
        [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx**2+qy**2)]
    ])
    return R @ np.array(point_cam) + np.array([tx, ty, tz])


def _find_gaps(nan_mask):
    """Find contiguous NaN gap regions. Returns list of (start, end) index tuples."""
    gaps = []
    in_gap = False
    start = 0
    for i, is_nan in enumerate(nan_mask):
        if is_nan and not in_gap:
            start = i
            in_gap = True
        elif not is_nan and in_gap:
            gaps.append((start, i))
            in_gap = False
    if in_gap:
        gaps.append((start, len(nan_mask)))
    return gaps


def interpolate_gaps(arr):
    """Gap-length-aware interpolation for [N,3] trajectory.

    Strategy per gap length:
    - Short gaps (<= 15 frames): CubicSpline (C2 smooth, best for short spans)
    - Medium gaps (16-30 frames): PCHIP (monotonic, no overshoot)
    - Long gaps (> 30 frames): Linear (safe, no wild oscillations)

    Applied per-axis independently. All valid points are used to fit the
    interpolator, giving global context even for local gap fills.
    """
    result = arr.copy()
    for ax in range(3):
        col = result[:, ax]
        valid_mask = ~np.isnan(col)
        if valid_mask.all() or not valid_mask.any():
            continue

        valid_idx = np.where(valid_mask)[0]
        valid_vals = col[valid_mask]

        # Identify each contiguous NaN gap
        nan_mask = np.isnan(col)
        gaps = _find_gaps(nan_mask)

        for gap_start, gap_end in gaps:
            gap_len = gap_end - gap_start
            gap_indices = np.arange(gap_start, gap_end)

            # Need at least 2 valid points for any interpolation
            if len(valid_idx) < 2:
                col[gap_indices] = valid_vals[0] if len(valid_vals) > 0 else 0.0
                continue

            if gap_len <= 15:
                # CubicSpline: C2 continuous, smooth acceleration
                interp = CubicSpline(valid_idx, valid_vals)
                col[gap_indices] = interp(gap_indices)
            elif gap_len <= 30:
                # PCHIP: C1 continuous, monotonicity-preserving, no overshoot
                interp = PchipInterpolator(valid_idx, valid_vals)
                col[gap_indices] = interp(gap_indices)
            else:
                # Linear: safe for long gaps (e.g., 138-frame gap)
                col[gap_indices] = np.interp(gap_indices, valid_idx, valid_vals)

    return result


def smooth_trajectory_savgol(traj, window_length=7, polyorder=3):
    """Savitzky-Golay smoothing for [N,3] trajectory.

    Zero-phase (no lag in batch mode). Preserves trajectory shape and
    grasp dwell positions better than bidirectional EMA.

    Parameters:
    - window_length=7: 0.7s at 10 FPS. Conservative — preserves dwell.
    - polyorder=3: Cubic — preserves velocity and acceleration shape.
    - mode='nearest': Repeats edge value for endpoint handling.
    """
    if len(traj) < window_length:
        return traj.copy() if isinstance(traj, np.ndarray) else np.array(traj)
    return savgol_filter(traj, window_length, polyorder, axis=0, mode='nearest')


def process_session(session):
    print(f"\n{'='*60}")
    print(f"Wrist 3D Reconstruction: {session}")
    print(f"{'='*60}")

    # Load retarget data
    ret_path = RETARGET / f"{session}_retargeted.json"
    with open(ret_path) as f:
        ret = json.load(f)
    ts = ret["timesteps"]

    # Find R3D file
    r3d_dir = RAW / session
    r3d_files = list(r3d_dir.glob("*.r3d"))
    if not r3d_files:
        print(f"  ERROR: No .r3d in {r3d_dir}"); return None
    r3d_path = r3d_files[0]

    # Load metadata once
    import liblzfse
    with zipfile.ZipFile(r3d_path) as z:
        meta = json.load(z.open("metadata"))
    print(f"  R3D: {meta['w']}x{meta['h']} @ {meta['fps']}fps, {len(meta['poses'])} poses")
    fps_ratio = meta["fps"] / 30.0  # R3D 60fps, our pipeline 30fps

    # TRK-05: Detect if HaMeR 3D wrist data is available
    has_hamer_3d = any(t.get("wrist_3d_camera") for t in ts)
    if has_hamer_3d:
        print(f"  Using HaMeR 3D wrist output (TRK-05: skip depth unprojection)")
    else:
        print(f"  Using depth-map unprojection (MediaPipe mode)")

    # Process each frame
    wrist_3d_world = []
    grasping = []
    failed = 0
    for i, t in enumerate(ts):
        wp = t.get("wrist_pixel")
        if wp is None and not t.get("wrist_3d_camera"):
            wrist_3d_world.append(None)
            grasping.append(t.get("grasping", False))
            failed += 1
            continue

        r3d_idx = min(int(i * fps_ratio), len(meta["poses"]) - 1)

        # TRK-05: Use HaMeR 3D wrist when available (skip depth lookup)
        if t.get("wrist_3d_camera"):
            point_cam = np.array(t["wrist_3d_camera"])
            pose_idx = min(r3d_idx, len(meta["poses"]) - 1)
            pose = meta["poses"][pose_idx]
            point = cam_to_world(point_cam, pose)
            wrist_3d_world.append(point.tolist())
            grasping.append(t.get("grasping", False))
            continue

        # Fallback: depth-map unprojection (MediaPipe path)
        if wp is None:
            wrist_3d_world.append(None)
            grasping.append(t.get("grasping", False))
            failed += 1
            continue

        # Load depth for this frame
        try:
            _, depth = load_r3d_depth(r3d_path, r3d_idx)
        except (KeyError, Exception):
            wrist_3d_world.append(None)
            grasping.append(t.get("grasping", False))
            failed += 1
            continue

        point = pixel_to_world(wp[0], wp[1], depth, meta, r3d_idx)
        wrist_3d_world.append(point.tolist() if point is not None else None)
        grasping.append(t.get("grasping", False))
        if point is None:
            failed += 1

    valid = sum(1 for w in wrist_3d_world if w is not None)
    print(f"  Reconstructed: {valid}/{len(ts)} frames ({failed} failed)")

    # Convert to array with NaN for missing frames
    arr = []
    for w in wrist_3d_world:
        arr.append(w if w is not None else [np.nan, np.nan, np.nan])
    arr = np.array(arr, dtype=float)

    # TRK-01: Gap-length-aware interpolation (cubic < 15, PCHIP 15-30, linear > 30)
    interpolated = interpolate_gaps(arr)
    assert not np.isnan(interpolated).any(), "NaN gaps remain after interpolation"

    # TRK-02: Savitzky-Golay spatial filter (zero-phase, preserves grasp dwell)
    smoothed = smooth_trajectory_savgol(interpolated, window_length=7, polyorder=3)

    # Velocity clamp: limit frame-to-frame displacement to 3cm
    MAX_STEP = 0.03  # 3cm per frame at 10fps = 0.3 m/s
    for i in range(1, len(smoothed)):
        delta = smoothed[i] - smoothed[i-1]
        norm = np.linalg.norm(delta)
        if norm > MAX_STEP:
            smoothed[i] = smoothed[i-1] + delta * (MAX_STEP / norm)

    assert not np.isnan(smoothed).any(), "NaN values after smoothing"

    # Validate: check for large jumps
    disps = np.linalg.norm(np.diff(smoothed, axis=0), axis=1)
    max_jump = disps.max()
    n_large_jumps = (disps > 0.03).sum()
    if n_large_jumps > 0:
        print(f"  WARNING: {n_large_jumps} frames with >3cm jump (max={max_jump:.4f}m)")

    # TRK-03: Verify grasping signal remains binary
    unique_grasp = set(np.unique(np.array(grasping, dtype=float)).tolist())
    assert unique_grasp.issubset({0.0, 1.0}), \
        f"Grasping signal corrupted: unique values = {unique_grasp}"

    # Stats
    print(f"  World coords range:")
    for ax, name in enumerate("XYZ"):
        mn, mx = smoothed[:, ax].min(), smoothed[:, ax].max()
        print(f"    {name}: [{mn:.4f}, {mx:.4f}] (range={mx-mn:.4f})")

    # Frame-to-frame displacement
    disps = np.linalg.norm(np.diff(smoothed, axis=0), axis=1)
    print(f"  Displacement: mean={disps.mean():.4f} max={disps.max():.4f}")

    # Save
    result = {
        "session": session,
        "n_frames": len(ts),
        "n_valid": valid,
        "wrist_world_raw": [w if w is not None else None for w in wrist_3d_world],
        "wrist_world_smooth": smoothed.tolist(),
        "grasping": grasping,
        "world_range": {
            "x": [float(smoothed[:, 0].min()), float(smoothed[:, 0].max())],
            "y": [float(smoothed[:, 1].min()), float(smoothed[:, 1].max())],
            "z": [float(smoothed[:, 2].min()), float(smoothed[:, 2].max())],
        }
    }
    out_path = OUTPUT / f"{session}_wrist3d.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out_path}")
    return result


if __name__ == "__main__":
    sessions = sys.argv[1:] or ["stack1", "stack2", "picknplace1", "picknplace2", "sort2"]
    for s in sessions:
        try:
            process_session(s)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
