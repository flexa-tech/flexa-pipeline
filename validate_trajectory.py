#!/usr/bin/env python3
"""Multi-layer trajectory validator using Pinocchio.

Takes calibrated JSON from wrist_trajectories/ and produces a per-frame
pass/fail report covering IK feasibility, joint limits, dynamics, trajectory
quality, and task semantics.

Usage:
    python validate_trajectory.py wrist_trajectories/stack2_calibrated.json [--verbose] [--output report.json]

Importable:
    from validate_trajectory import validate
    report = validate("wrist_trajectories/stack2_calibrated.json")
"""

import argparse
import json
import numpy as np
import pinocchio as pin
from pathlib import Path

from pipeline_config import PROJECT_ROOT, G1_URDF

# ---------- constants (matching mujoco_g1_v10.py) ----------

URDF_PATH = G1_URDF

RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

SEED = np.array([0.0, 0.0, 0.0, 2.4, 0.0, -1.5, 0.0])

# End-effector and palm frames (matches MuJoCo script)
EE_FRAME = "right_wrist_yaw_link"
PALM_FRAMES = [
    "right_hand_thumb_2_link",
    "right_hand_index_1_link",
    "right_hand_middle_1_link",
]

# Workspace clamp (from mujoco_g1_v10.py)
WORKSPACE_MIN = np.array([0.10, -0.45, 0.80])
WORKSPACE_MAX = np.array([0.65,  0.30, 1.20])
G1_TABLE_HEIGHT = 0.78

FPS = 30  # pipeline standard

# IK parameters
IK_MAX_ITER = 200
IK_CONVERGE_THRESH = 0.004   # 4mm — matches MuJoCo IK
IK_DAMPING = 0.005
IK_NULL_GAIN = 0.05

# Thresholds
IK_ERROR_WARN = 0.02         # 2cm — warn threshold (matches mujoco_g1_v10.py)
JOINT_LIMIT_MARGIN = 0.01    # rad margin before flagging
MANIPULABILITY_THRESH = 1e-5  # near-singularity threshold
ACCEL_LIMIT = 50.0           # rad/s^2 reasonable bound
VEL_LIMIT = 10.0             # rad/s conservative default
GRASP_RADIUS = 0.25          # 25cm proximity for grasp onset (accounts for IK residual + noise)
POST_GRASP_Z_DROP = 0.10     # max allowed Z drop after grasp (10cm, Z-floor clamping artifact)


def _load_model():
    """Load Pinocchio model from URDF."""
    model = pin.buildModelFromUrdf(str(URDF_PATH))
    data = model.createData()
    return model, data


def _get_arm_indices(model):
    """Get joint IDs and q/v-indices for the 7 right-arm joints."""
    joint_ids = []
    q_indices = []
    v_indices = []
    for jname in RIGHT_ARM_JOINTS:
        jid = model.getJointId(jname)
        joint_ids.append(jid)
        q_indices.append(model.joints[jid].idx_q)
        v_indices.append(model.joints[jid].idx_v)
    return joint_ids, np.array(q_indices), np.array(v_indices)


def _stand_config(model):
    """Return the standing configuration with floating base at origin.

    The URDF already encodes the pelvis body offset (z=0.793),
    so the floating base transform should be identity.
    """
    q = pin.neutral(model)
    q[0:3] = [0, 0, 0]
    q[3:7] = [0, 0, 0, 1]  # identity quaternion (x,y,z,w)
    return q


def _palm_center(model, data, palm_frame_ids):
    """Compute average position of palm frames (matches MuJoCo palm_center)."""
    pts = [data.oMf[fid].translation.copy() for fid in palm_frame_ids]
    return np.mean(pts, axis=0)


def _ik_solve_palm(model, data, q_full, target_pos,
                   q_indices, v_indices, ee_frame_id, palm_frame_ids,
                   warm_start=None):
    """Palm-center-aware damped-least-squares IK for the 7-DOF right arm.

    Matches the MuJoCo IK approach: compute palm center, derive wrist target
    from (target - palm_to_wrist_offset), then solve IK on the wrist frame.

    Args:
        model: Pinocchio model
        data: Pinocchio data
        q_full: Full configuration vector (modified in-place for arm joints)
        target_pos: 3D target position for palm center
        q_indices: Indices into q for arm joints
        v_indices: Indices into v for arm joints
        ee_frame_id: Frame ID for wrist (right_wrist_yaw_link)
        palm_frame_ids: Frame IDs for palm bodies
        warm_start: Optional 7-element joint angle array to initialize from

    Returns:
        q_arm: 7-element joint angle vector
        error: Palm-center position error magnitude
        converged: Whether IK converged
        palm_pos: Final palm center position
    """
    # Initialize from warm start or SEED
    init_q = warm_start if warm_start is not None else SEED
    for i, qi in enumerate(q_indices):
        q_full[qi] = init_q[i]

    for iteration in range(IK_MAX_ITER):
        pin.forwardKinematics(model, data, q_full)
        pin.updateFramePlacements(model, data)

        # Compute palm center and wrist-to-palm offset
        pc = _palm_center(model, data, palm_frame_ids)
        wrist_pos = data.oMf[ee_frame_id].translation.copy()
        w2p = pc - wrist_pos  # palm center offset from wrist

        # Derive wrist target so that palm center lands on target_pos
        wrist_target = target_pos - w2p
        err = wrist_target - wrist_pos
        err_norm = np.linalg.norm(err)

        if err_norm < IK_CONVERGE_THRESH:
            q_arm = q_full[q_indices].copy()
            palm_err = np.linalg.norm(target_pos - pc)
            return q_arm, palm_err, True, pc.copy()

        # Compute Jacobian at wrist frame
        J_full = pin.computeFrameJacobian(model, data, q_full, ee_frame_id,
                                          pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J = J_full[:3, v_indices]

        # Damped least squares
        JJT = J @ J.T + IK_DAMPING * np.eye(3)
        dq = J.T @ np.linalg.solve(JJT, err)

        # Null-space bias toward SEED
        null = np.eye(len(q_indices)) - J.T @ np.linalg.solve(JJT, J)
        q_arm_cur = q_full[q_indices].copy()
        dq += null @ (SEED - q_arm_cur) * IK_NULL_GAIN

        # Step with adaptive gain
        step = min(1.0, 0.25 + err_norm)
        for i, qi in enumerate(q_indices):
            q_full[qi] += dq[i] * step

    # Did not converge — compute final error
    pin.forwardKinematics(model, data, q_full)
    pin.updateFramePlacements(model, data)
    pc = _palm_center(model, data, palm_frame_ids)
    palm_err = np.linalg.norm(target_pos - pc)
    q_arm = q_full[q_indices].copy()
    return q_arm, palm_err, False, pc.copy()


def validate(json_path: str, verbose: bool = False) -> dict:
    """Run multi-layer trajectory validation.

    Args:
        json_path: Path to calibrated JSON file
        verbose: Print per-frame details

    Returns:
        Report dictionary with summary, per_frame, trajectory_quality, task_semantics
    """
    # Load data
    calib = json.loads(Path(json_path).read_text())
    wrist = np.array(calib["wrist_sim"], dtype=float)
    grasping = np.array(calib["grasping"])
    objects_sim = calib["objects_sim"]
    n_frames = len(wrist)

    # Parse object positions
    if isinstance(objects_sim, dict):
        obj_positions = {k: np.array(v[:3]) for k, v in objects_sim.items()}
    else:
        obj_names = [f"obj_{i}" for i in range(len(objects_sim))]
        obj_positions = {n: np.array(p[:3]) for n, p in zip(obj_names, objects_sim)}

    # Load Pinocchio model
    model, data = _load_model()
    joint_ids, q_indices, v_indices = _get_arm_indices(model)
    ee_frame_id = model.getFrameId(EE_FRAME)
    palm_frame_ids = [model.getFrameId(fn) for fn in PALM_FRAMES]
    q_full = _stand_config(model)

    # Get joint limits for arm joints
    lower_limits = model.lowerPositionLimit[q_indices]
    upper_limits = model.upperPositionLimit[q_indices]

    if verbose:
        print(f"  Model: nq={model.nq}, nv={model.nv}")
        print(f"  Arm q_indices: {q_indices.tolist()}")
        print(f"  Joint limits:")
        for j, jn in enumerate(RIGHT_ARM_JOINTS):
            print(f"    {jn}: [{lower_limits[j]:.4f}, {upper_limits[j]:.4f}]")

    # ===== Per-frame processing =====
    per_frame = []
    all_q = np.zeros((n_frames, 7))
    all_palm = np.zeros((n_frames, 3))
    ik_converged_count = 0
    joint_limit_violations = 0
    manipulability_warnings = 0
    warm_q = SEED.copy()

    for i in range(n_frames):
        frame_report = {"frame": i}

        # Prepare target (with workspace clamping, same as mujoco_g1_v10.py)
        target = wrist[i].copy()
        target[2] = max(target[2], G1_TABLE_HEIGHT + 0.02)
        target = np.clip(target, WORKSPACE_MIN, WORKSPACE_MAX)

        # Layer 1: IK solve (palm-center aware, matching MuJoCo approach)
        q_arm, ik_err, converged, palm_pos = _ik_solve_palm(
            model, data, q_full, target,
            q_indices, v_indices, ee_frame_id, palm_frame_ids,
            warm_start=warm_q
        )
        all_q[i] = q_arm
        all_palm[i] = palm_pos
        warm_q = q_arm.copy()  # warm-start next frame

        frame_report["ik_error"] = float(ik_err)
        frame_report["ik_converged"] = converged
        if converged:
            ik_converged_count += 1

        # Layer 2a: Joint limits check
        below = q_arm < (lower_limits + JOINT_LIMIT_MARGIN)
        above = q_arm > (upper_limits - JOINT_LIMIT_MARGIN)
        limits_ok = not (below.any() or above.any())
        frame_report["joint_limits_ok"] = limits_ok
        if not limits_ok:
            joint_limit_violations += 1
            violations = []
            for j in range(7):
                if below[j]:
                    violations.append({"joint": RIGHT_ARM_JOINTS[j], "value": float(q_arm[j]),
                                      "limit": float(lower_limits[j]), "side": "lower"})
                if above[j]:
                    violations.append({"joint": RIGHT_ARM_JOINTS[j], "value": float(q_arm[j]),
                                      "limit": float(upper_limits[j]), "side": "upper"})
            frame_report["joint_limit_details"] = violations

        # Layer 2b: Manipulability
        J_full = pin.computeFrameJacobian(model, data, q_full, ee_frame_id,
                                          pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J = J_full[:3, v_indices]
        manip = np.sqrt(max(0, np.linalg.det(J @ J.T)))
        frame_report["manipulability"] = float(manip)
        if manip < MANIPULABILITY_THRESH:
            manipulability_warnings += 1
            frame_report["near_singular"] = True

        per_frame.append(frame_report)

        if verbose and i % 50 == 0:
            print(f"  Frame {i:4d}/{n_frames}: IK err={ik_err:.4f}m  "
                  f"converged={converged}  manip={manip:.6f}  limits_ok={limits_ok}  "
                  f"palm={palm_pos.round(3)}")

    # ===== Layer 3: Dynamic checks (frame-to-frame) =====
    velocity_violations = 0
    accel_violations = 0

    if n_frames > 1:
        dq = np.diff(all_q, axis=0) * FPS  # joint velocities (rad/s)
        for i in range(len(dq)):
            per_frame[i + 1]["max_joint_vel"] = float(np.max(np.abs(dq[i])))

        if n_frames > 2:
            ddq = np.diff(dq, axis=0) * FPS  # joint accelerations (rad/s^2)
            for i in range(len(ddq)):
                max_accel = float(np.max(np.abs(ddq[i])))
                per_frame[i + 2]["max_joint_accel"] = max_accel
                if max_accel > ACCEL_LIMIT:
                    accel_violations += 1
                    per_frame[i + 2]["accel_exceeded"] = True

        # Check velocity limits
        for i in range(len(dq)):
            if np.max(np.abs(dq[i])) > VEL_LIMIT:
                velocity_violations += 1
                per_frame[i + 1]["vel_exceeded"] = True

    # ===== Layer 4: Trajectory quality =====
    trajectory_quality = {}

    if n_frames > 3:
        # Smoothness: RMS jerk per joint
        d3q = np.diff(all_q, n=3, axis=0) * (FPS ** 3)
        rms_jerk = np.sqrt(np.mean(d3q ** 2, axis=0))
        trajectory_quality["rms_jerk_per_joint"] = {
            RIGHT_ARM_JOINTS[j]: float(rms_jerk[j]) for j in range(7)
        }
        trajectory_quality["mean_rms_jerk"] = float(np.mean(rms_jerk))

    # Path efficiency: straight-line / arc-length ratio of palm trajectory
    if n_frames > 1:
        segments = np.linalg.norm(np.diff(all_palm, axis=0), axis=1)
        arc_length = float(np.sum(segments))
        straight = float(np.linalg.norm(all_palm[-1] - all_palm[0]))
        trajectory_quality["palm_arc_length_m"] = arc_length
        trajectory_quality["palm_straight_m"] = straight
        trajectory_quality["path_efficiency"] = straight / arc_length if arc_length > 0 else 0.0

    # ===== Layer 5: Task semantics =====
    task_semantics = {}

    # Find grasp transitions
    grasp_bool = np.array([bool(g) for g in grasping])
    grasp_onsets = []
    grasp_offsets = []
    for i in range(1, n_frames):
        if grasp_bool[i] and not grasp_bool[i - 1]:
            grasp_onsets.append(i)
        if not grasp_bool[i] and grasp_bool[i - 1]:
            grasp_offsets.append(i)

    task_semantics["grasp_onsets"] = grasp_onsets
    task_semantics["grasp_offsets"] = grasp_offsets
    task_semantics["total_grasp_frames"] = int(grasp_bool.sum())

    # 5a: Grasp proximity -- at grasp onset, is palm near an object?
    if grasp_onsets:
        onset_frame = grasp_onsets[0]
        palm_at_onset = all_palm[onset_frame]
        min_dist = min(np.linalg.norm(palm_at_onset - pos) for pos in obj_positions.values())
        task_semantics["grasp_onset_frame"] = onset_frame
        task_semantics["grasp_onset_palm_obj_dist"] = float(min_dist)
        task_semantics["grasp_proximity_ok"] = min_dist < GRASP_RADIUS
    else:
        task_semantics["grasp_proximity_ok"] = None
        task_semantics["note"] = "No grasp onset detected"

    # 5b: Post-grasp lift -- Z should not drop significantly after grasp
    if grasp_onsets:
        onset = grasp_onsets[0]
        grasp_frames = [i for i in range(onset, n_frames) if grasp_bool[i]]
        if len(grasp_frames) > 5:
            z_at_onset = all_palm[onset, 2]
            z_min_post = np.min(all_palm[grasp_frames, 2])
            z_drop = z_at_onset - z_min_post
            task_semantics["post_grasp_z_drop"] = float(z_drop)
            task_semantics["post_grasp_lift_ok"] = z_drop < POST_GRASP_Z_DROP
        else:
            task_semantics["post_grasp_lift_ok"] = None
    else:
        task_semantics["post_grasp_lift_ok"] = None

    # ===== Build summary =====
    ik_errors = [f["ik_error"] for f in per_frame]
    ik_arr = np.array(ik_errors)

    # Tiered IK error buckets
    ik_tiers = {
        "lt_5mm": int(np.sum(ik_arr < 0.005)),
        "5mm_to_2cm": int(np.sum((ik_arr >= 0.005) & (ik_arr < 0.02))),
        "2cm_to_5cm": int(np.sum((ik_arr >= 0.02) & (ik_arr < 0.05))),
        "5cm_to_10cm": int(np.sum((ik_arr >= 0.05) & (ik_arr < 0.10))),
        "gt_10cm": int(np.sum(ik_arr >= 0.10)),
    }

    summary = {
        "total_frames": n_frames,
        "ik_converged": ik_converged_count,
        "ik_failed": n_frames - ik_converged_count,
        "ik_warn_gt_2cm": int(np.sum(ik_arr > IK_ERROR_WARN)),
        "ik_error_tiers": ik_tiers,
        "ik_error_mean": float(np.mean(ik_errors)),
        "ik_error_max": float(np.max(ik_errors)),
        "ik_error_p50": float(np.percentile(ik_errors, 50)),
        "ik_error_p95": float(np.percentile(ik_errors, 95)),
        "joint_limit_violations": joint_limit_violations,
        "manipulability_warnings": manipulability_warnings,
        "velocity_violations": velocity_violations,
        "accel_violations": accel_violations,
    }

    report = {
        "summary": summary,
        "per_frame": per_frame,
        "trajectory_quality": trajectory_quality,
        "task_semantics": task_semantics,
    }

    return report


def print_report(report: dict):
    """Print human-readable summary."""
    s = report["summary"]
    tq = report["trajectory_quality"]
    ts = report["task_semantics"]

    print("\n" + "=" * 60)
    print("  TRAJECTORY VALIDATION REPORT")
    print("=" * 60)

    # Layer 1: IK
    ik_pct = s["ik_converged"] / s["total_frames"] * 100
    # Consider IK "useful" if error < 5cm (within IK residual for workspace-limited targets)
    ik_useful = s["ik_error_tiers"]["lt_5mm"] + s["ik_error_tiers"]["5mm_to_2cm"] + s["ik_error_tiers"]["2cm_to_5cm"]
    ik_useful_pct = ik_useful / s["total_frames"] * 100
    print(f"\n  Layer 1 -- IK Feasibility:     {'PASS' if ik_useful_pct > 50 else 'WARN'}")
    print(f"    Converged (<{IK_CONVERGE_THRESH*1000:.0f}mm): "
          f"{s['ik_converged']}/{s['total_frames']} ({ik_pct:.1f}%)")
    print(f"    Usable (<5cm):  {ik_useful}/{s['total_frames']} ({ik_useful_pct:.1f}%)")
    t = s["ik_error_tiers"]
    print(f"    Tiers:  <5mm={t['lt_5mm']}  5mm-2cm={t['5mm_to_2cm']}  "
          f"2cm-5cm={t['2cm_to_5cm']}  5cm-10cm={t['5cm_to_10cm']}  >10cm={t['gt_10cm']}")
    print(f"    Error:  mean={s['ik_error_mean']:.4f}m  p50={s['ik_error_p50']:.4f}m  "
          f"p95={s['ik_error_p95']:.4f}m  max={s['ik_error_max']:.4f}m")

    # Layer 2: Kinematics
    limits_ok = s["joint_limit_violations"] == 0
    manip_ok = s["manipulability_warnings"] == 0
    print(f"\n  Layer 2 -- Kinematic Checks:   {'PASS' if limits_ok and manip_ok else 'WARN'}")
    print(f"    Joint limit violations: {s['joint_limit_violations']}")
    print(f"    Manipulability warnings: {s['manipulability_warnings']}")

    # Layer 3: Dynamics
    vel_ok = s["velocity_violations"] == 0
    acc_ok = s["accel_violations"] == 0
    print(f"\n  Layer 3 -- Dynamic Checks:     {'PASS' if vel_ok and acc_ok else 'WARN'}")
    print(f"    Velocity violations (>{VEL_LIMIT} rad/s): {s['velocity_violations']}")
    print(f"    Acceleration violations (>{ACCEL_LIMIT} rad/s^2): {s['accel_violations']}")

    # Layer 4: Quality
    print(f"\n  Layer 4 -- Trajectory Quality:")
    if "path_efficiency" in tq:
        print(f"    Path efficiency: {tq['path_efficiency']:.3f}")
        print(f"    Palm arc length: {tq['palm_arc_length_m']:.3f}m, "
              f"Straight: {tq['palm_straight_m']:.3f}m")
    if "mean_rms_jerk" in tq:
        print(f"    Mean RMS jerk: {tq['mean_rms_jerk']:.1f} rad/s^3")

    # Layer 5: Task semantics
    print(f"\n  Layer 5 -- Task Semantics:")
    print(f"    Grasp frames: {ts.get('total_grasp_frames', 0)}")
    if ts.get("grasp_proximity_ok") is not None:
        ok = ts["grasp_proximity_ok"]
        print(f"    Grasp proximity:  {'PASS' if ok else 'FAIL'}  "
              f"(dist={ts.get('grasp_onset_palm_obj_dist', 0):.3f}m, thresh={GRASP_RADIUS}m)")
    else:
        print(f"    Grasp proximity:  N/A ({ts.get('note', 'no grasp data')})")

    if ts.get("post_grasp_lift_ok") is not None:
        ok = ts["post_grasp_lift_ok"]
        print(f"    Post-grasp lift:  {'PASS' if ok else 'FAIL'}  "
              f"(z_drop={ts.get('post_grasp_z_drop', 0):.3f}m)")
    else:
        print(f"    Post-grasp lift:  N/A")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Validate calibrated wrist trajectory using Pinocchio.")
    parser.add_argument("json_path", help="Path to calibrated JSON file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-frame details")
    parser.add_argument("--output", "-o", help="Write JSON report to file")
    args = parser.parse_args()

    print(f"Validating: {args.json_path}")
    report = validate(args.json_path, verbose=args.verbose)

    print_report(report)

    if args.output:
        out_report = report.copy()
        if not args.verbose:
            out_report["per_frame"] = f"[{len(report['per_frame'])} frames -- use --verbose to include]"
        Path(args.output).write_text(json.dumps(out_report, indent=2, default=str))
        print(f"\nReport written to {args.output}")
    else:
        out_path = Path(args.json_path).with_suffix(".validation.json")
        Path(out_path).write_text(json.dumps(report, indent=2, default=str))
        print(f"\nFull report written to {out_path}")


if __name__ == "__main__":
    main()
