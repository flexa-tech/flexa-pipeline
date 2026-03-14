#!/usr/bin/env python3
"""G1 v10: Contact-based physics grasping.

- Arm joints set kinematically via IK
- Block grasping uses pure contact friction (no kinematic attachment)
- Fingers physically close around block; MuJoCo friction solver holds it
- Support block pinned for stability; pick block fully dynamic

Output: sim_renders/<task>_g1_v10.mp4
"""

import mujoco
import numpy as np
import json, sys, tempfile, subprocess, shutil, os
from pathlib import Path
from PIL import Image

from pipeline_config import G1_DIR as MENAGERIE, CALIB_DIR, OUT_DIR

BLOCK_HALF = 0.03  # 6cm cubes
G1_TABLE_HEIGHT = 0.78
RENDER_W, RENDER_H = 640, 480
FPS = 10
TIMESTEP = 0.002
SUBSTEPS = max(25, min(50, int(1.0 / (FPS * TIMESTEP))))

RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

PALM_BODIES = ["right_hand_thumb_2_link", "right_hand_index_1_link", "right_hand_middle_1_link"]
SEED = np.array([0.0, 0.0, 0.0, 2.4, 0.0, -1.5, 0.0])

HAND_CTRL_START = 36
FINGER_OPEN = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
FINGER_CLOSED = np.array([0.48, -0.60, -0.72, 0.96, 1.08, 0.96, 1.08])  # tighter wrap for contact grip (~1.2x)
FINGER_PRESHAPE = np.array([0.2, -0.25, -0.3, 0.4, 0.45, 0.4, 0.45])  # 50% of CLOSED — anticipatory curl

# Distance-based pre-shaping thresholds (OUT-03)
PRESHAPE_DIST_START = 0.20   # begin finger curl at 20cm from nearest block
PRESHAPE_DIST_FULL  = 0.06   # full pre-shape at 6cm (approximately block diameter)

FINGER_GAIN_MULTIPLIER = 40.0


# Reachable workspace envelope for IK target clamping (RET-03)
# Derived from G1 right arm reach + table geometry
WORKSPACE_MIN = np.array([0.10, -0.45, 0.80])   # X_min, Y_min, Z_min
WORKSPACE_MAX = np.array([0.65,  0.30, 1.20])   # X_max, Y_max, Z_max


def debounce_grasping(grasping, min_on=25, on_thresh=3, off_thresh=20, merge_gap=150):
    """Debounce noisy grasping signal into clean grip/release cycles.

    Two-pass approach:
    1. State-machine debounce: hysteresis with on/off thresholds + min duration
    2. Merge nearby events: bridge gaps shorter than merge_gap frames

    MediaPipe's ~52% detection rate produces scattered single-frame grasping
    estimates. This smooths them into 1-2 sustained grip events.
    """
    g = np.array(grasping, dtype=float)
    n = len(g)
    result = np.zeros(n, dtype=float)
    state = False
    count = 0
    on_start = 0

    # Pass 1: State-machine debounce
    for i in range(n):
        if state:
            if g[i] < 0.5:
                count += 1
                if count >= off_thresh and (i - on_start) >= min_on:
                    state = False
                    count = 0
            else:
                count = 0
        else:
            if g[i] > 0.5:
                count += 1
                if count >= on_thresh:
                    state = True
                    on_start = i - count + 1
                    result[max(0, i - count + 1):i + 1] = 1.0
                    count = 0
            else:
                count = 0
        if state:
            result[i] = 1.0

    # Pass 2: Merge grip events separated by < merge_gap frames
    segments = []
    in_seg = False
    seg_start = 0
    for i in range(n):
        if result[i] > 0.5 and not in_seg:
            seg_start = i
            in_seg = True
        elif result[i] < 0.5 and in_seg:
            segments.append((seg_start, i - 1))
            in_seg = False
    if in_seg:
        segments.append((seg_start, n - 1))

    if len(segments) > 1:
        merged = [segments[0]]
        for s, e in segments[1:]:
            prev_s, prev_e = merged[-1]
            if s - prev_e <= merge_gap:
                merged[-1] = (prev_s, e)
            else:
                merged.append((s, e))
        result = np.zeros(n, dtype=float)
        for s, e in merged:
            result[s:e + 1] = 1.0

    return result


def scene_xml(obj_xml: str) -> str:
    return f"""<mujoco>
  <include file="g1_with_hands.xml"/>
  <option timestep="{TIMESTEP}" gravity="0 0 -9.81" cone="elliptic" impratio="10"/>
  <visual><global offwidth="{RENDER_W}" offheight="{RENDER_H}"/></visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
             rgb1="0.15 0.2 0.25" rgb2="0.1 0.15 0.2" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="0 0 2.5" dir="0 0 -1" directional="true"/>
    <light pos="1.5 -0.5 2" dir="-0.5 0.3 -1" diffuse="0.4 0.4 0.4"/>

    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" contype="1" conaffinity="1"/>

    <body name="table" pos="0.32 0 {G1_TABLE_HEIGHT/2}">
      <geom type="box" size="0.4 0.6 {G1_TABLE_HEIGHT/2}" rgba="0.55 0.35 0.18 1" mass="50"
            friction="1.0 0.005 0.0001" contype="1" conaffinity="1"/>
    </body>

    {obj_xml}

    <camera name="front" pos="1.4 0.9 1.1" xyaxes="-0.55 0.84 0.0 -0.20 -0.13 0.97" fovy="50"/>
  </worldbody>
</mujoco>"""


def jid(m, name, t):
    return mujoco.mj_name2id(m, t, name)


def palm_center(model, data):
    pts = []
    for bname in PALM_BODIES:
        bid = jid(model, bname, mujoco.mjtObj.mjOBJ_BODY)
        pts.append(data.xpos[bid].copy())
    return np.mean(pts, axis=0)


def ik_solve(model, data, target, arm_qa, arm_da, max_iter=150):
    wrist_bid = jid(model, "right_wrist_yaw_link", mujoco.mjtObj.mjOBJ_BODY)
    palm_bids = [jid(model, b, mujoco.mjtObj.mjOBJ_BODY) for b in PALM_BODIES]
    jac = np.zeros((3, model.nv))
    jac_r = np.zeros((3, model.nv))
    for _ in range(max_iter):
        mujoco.mj_forward(model, data)
        pc = np.mean([data.xpos[bid] for bid in palm_bids], axis=0)
        w2p = pc - data.xpos[wrist_bid]
        wrist_target = target - w2p
        err = wrist_target - data.xpos[wrist_bid]
        if np.linalg.norm(err) < 0.004:
            break
        mujoco.mj_jacBody(model, data, jac, jac_r, wrist_bid)
        J = jac[:, arm_da]
        JJT = J @ J.T + 0.005 * np.eye(3)
        dq = J.T @ np.linalg.solve(JJT, err)
        null = np.eye(len(arm_qa)) - J.T @ np.linalg.solve(JJT, J)
        qcur = np.array([data.qpos[a] for a in arm_qa])
        dq += null @ (SEED - qcur) * 0.05
        step = min(1.0, 0.25 + np.linalg.norm(err))
        for i, a in enumerate(arm_qa):
            data.qpos[a] += dq[i] * step
    mujoco.mj_forward(model, data)
    return np.array([data.qpos[a] for a in arm_qa])


def build_objects(obj_names):
    colors = {"block_a": "0.95 0.15 0.15 1", "block_b": "0.15 0.25 0.95 1"}
    parts = []
    for nm in obj_names:
        c = colors.get(nm, "0.2 0.8 0.3 1")
        parts.append(
            f'<body name="{nm}" pos="0 0 0">'
            f'<freejoint name="{nm}_jnt"/>'
            f'<geom type="box" size="{BLOCK_HALF} {BLOCK_HALF} {BLOCK_HALF}" rgba="{c}" mass="0.3" '
            f'friction="3.0 0.02 0.002" condim="4" contype="1" conaffinity="1"/>'
            f'</body>'
        )
    return "\n    ".join(parts)


def render_task(task_name: str):
    calib = json.loads((CALIB_DIR / f"{task_name}_calibrated.json").read_text())
    wrist = np.array(calib["wrist_sim"], dtype=float)
    grasping_raw = np.array(calib["grasping"], dtype=float)
    grasping = debounce_grasping(grasping_raw)
    n_raw = int(grasping_raw.sum())
    n_debounced = int(grasping.sum())
    print(f"  Grasping debounce: {n_raw} raw → {n_debounced} debounced frames")
    objects_raw = calib["objects_sim"]

    if isinstance(objects_raw, dict):
        obj_names = list(objects_raw.keys())
        obj_xy = {k: np.array(v[:2]) for k, v in objects_raw.items()}
    else:
        obj_names = ["block_a", "block_b"] if len(objects_raw) >= 2 else [f"obj_{chr(97+i)}" for i in range(len(objects_raw))]
        obj_xy = {n: np.array(p[:2]) for n, p in zip(obj_names, objects_raw)}

    xml = scene_xml(build_objects(obj_names))
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False, dir=str(MENAGERIE))
    tmp.write(xml); tmp.close()

    model = mujoco.MjModel.from_xml_path(tmp.name)
    data = mujoco.MjData(model)

    # Increase finger actuator gains
    for ai in range(min(model.nu, HAND_CTRL_START + 7)):
        if ai >= HAND_CTRL_START:
            model.actuator_gainprm[ai, 0] *= FINGER_GAIN_MULTIPLIER
            model.actuator_biasprm[ai, 1] *= FINGER_GAIN_MULTIPLIER

    # Set high friction + torsional/rolling on right-hand finger geoms for contact grip
    for gi in range(model.ngeom):
        bid = model.geom_bodyid[gi]
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if bname.startswith("right_hand_"):
            model.geom_friction[gi] = [3.0, 0.02, 0.002]
            model.geom_condim[gi] = 4

    key_id = jid(model, "stand", mujoco.mjtObj.mjOBJ_KEY)
    mujoco.mj_resetDataKeyframe(model, data, key_id)

    arm_qa, arm_da = [], []
    for jnm in RIGHT_ARM_JOINTS:
        j = jid(model, jnm, mujoco.mjtObj.mjOBJ_JOINT)
        arm_qa.append(model.jnt_qposadr[j])
        arm_da.append(model.jnt_dofadr[j])

    obj_jids = [jid(model, f"{nm}_jnt", mujoco.mjtObj.mjOBJ_JOINT) for nm in obj_names]
    obj_qadr = [model.jnt_qposadr[j] for j in obj_jids]
    obj_body_ids = [jid(model, nm, mujoco.mjtObj.mjOBJ_BODY) for nm in obj_names]

    # Derive block positions from wrist trajectory during grip phases
    table_x_offset = 0.32 - 0.5
    n = len(wrist)
    grip_idx = np.where(grasping > 0)[0]

    if len(grip_idx) >= 10:
        # Pick position: where the hand is at GRIP ONSET (first 15% of grip)
        onset_count = max(5, len(grip_idx) // 7)
        grip_onset = grip_idx[:onset_count]
        desired_pick_xy = wrist[grip_onset, :2].mean(axis=0)

        # Place position: where the hand is at GRIP END (last 15% of grip)
        grip_offset = grip_idx[-onset_count:]
        desired_place_xy = wrist[grip_offset, :2].mean(axis=0)

        # Support block goes at the place position (block gets stacked on it)
        desired_support_xy = desired_place_xy.copy()

        # Ensure minimum separation so blocks don't overlap at start
        sep = np.linalg.norm(desired_pick_xy - desired_support_xy)
        min_sep = 4 * BLOCK_HALF  # 12cm
        if sep < min_sep:
            direction = desired_support_xy - desired_pick_xy
            if np.linalg.norm(direction) > 0.01:
                direction = direction / np.linalg.norm(direction) * min_sep
            else:
                direction = np.array([0.0, -min_sep])
            desired_support_xy = desired_pick_xy + direction
    else:
        # No grip data — fall back to wrist midpoint
        desired_pick_xy = wrist[n // 4, :2].copy()
        desired_support_xy = wrist[3 * n // 4, :2].copy()

    # Clamp to reachable table area
    table_xy_min = np.array([0.15, -0.40])
    table_xy_max = np.array([0.55, 0.25])
    desired_pick_xy = np.clip(desired_pick_xy, table_xy_min, table_xy_max)
    desired_support_xy = np.clip(desired_support_xy, table_xy_min, table_xy_max)

    print(f"  Block placement from wrist trajectory:")
    print(f"    Pick block at:    XY=({desired_pick_xy[0]:.3f}, {desired_pick_xy[1]:.3f})")
    print(f"    Support block at: XY=({desired_support_xy[0]:.3f}, {desired_support_xy[1]:.3f})")
    print(f"    Separation: {np.linalg.norm(desired_pick_xy - desired_support_xy):.3f}m")

    # Determine which calibrated object is pick vs support
    # (pick = closer to grip onset wrist, support = other)
    pick_nm = min(obj_names, key=lambda nm: np.linalg.norm(obj_xy[nm] - desired_pick_xy))
    support_nm = [x for x in obj_names if x != pick_nm][0] if len(obj_names) >= 2 else None

    # Compute late grip window for compatibility with downstream code
    late = grip_idx[grip_idx >= int(0.6 * n)]
    if len(late) < 3:
        late = grip_idx[-max(1, len(grip_idx) // 2):] if len(grip_idx) else np.array([n - 10])
    ls, le = int(late[0]), int(min(late[-1] + 5, n - 1))

    # Place blocks at trajectory-derived positions
    obj_xy[pick_nm] = desired_pick_xy - np.array([table_x_offset, 0.0])
    if support_nm is not None:
        obj_xy[support_nm] = desired_support_xy - np.array([table_x_offset, 0.0])

    for nm, qa in zip(obj_names, obj_qadr):
        xy = obj_xy[nm]
        data.qpos[qa:qa+3] = [xy[0] + table_x_offset, xy[1], G1_TABLE_HEIGHT + BLOCK_HALF + 0.01]
        data.qpos[qa+3:qa+7] = [1, 0, 0, 0]

    for qa, v in zip(arm_qa, SEED):
        data.qpos[qa] = v

    block_geom_ids = set()
    for nm in obj_names:
        bid = jid(model, nm, mujoco.mjtObj.mjOBJ_BODY)
        for gi in range(model.ngeom):
            if model.geom_bodyid[gi] == bid:
                block_geom_ids.add(gi)

    arm_joint_set = set(RIGHT_ARM_JOINTS)
    obj_joint_set = set([f"{nm}_jnt" for nm in obj_names])

    mujoco.mj_forward(model, data)
    standing_qpos = data.qpos.copy()

    freeze_q_slices = []
    freeze_v_slices = []
    for jn in range(model.njnt):
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if not jname:
            continue
        if jname in arm_joint_set or jname in obj_joint_set:
            continue
        qa = model.jnt_qposadr[jn]
        da = model.jnt_dofadr[jn]
        if model.jnt_type[jn] == mujoco.mjtJoint.mjJNT_FREE:
            freeze_q_slices.append((qa, 7))
            freeze_v_slices.append((da, 6))
        else:
            freeze_q_slices.append((qa, 1))
            freeze_v_slices.append((da, 1))

    def freeze_robot():
        for qa, cnt in freeze_q_slices:
            data.qpos[qa:qa+cnt] = standing_qpos[qa:qa+cnt]
        for da, cnt in freeze_v_slices:
            data.qvel[da:da+cnt] = 0

    # 2-phase settle: first settle robot without blocks, then add blocks
    block_saved = {}
    for gi in block_geom_ids:
        block_saved[gi] = (int(model.geom_contype[gi]), int(model.geom_conaffinity[gi]))
        model.geom_contype[gi] = 0
        model.geom_conaffinity[gi] = 0

    for _ in range(200):
        freeze_robot()
        mujoco.mj_step(model, data)

    for gi, (ct, ca) in block_saved.items():
        model.geom_contype[gi] = ct
        model.geom_conaffinity[gi] = ca

    for nm, qa in zip(obj_names, obj_qadr):
        xy = obj_xy[nm]
        data.qpos[qa:qa+3] = [xy[0] + table_x_offset, xy[1], G1_TABLE_HEIGHT + BLOCK_HALF + 0.01]
        data.qpos[qa+3:qa+7] = [1, 0, 0, 0]
        da = model.jnt_dofadr[obj_jids[obj_names.index(nm)]]
        data.qvel[da:da+6] = 0

    mujoco.mj_forward(model, data)
    for _ in range(200):
        freeze_robot()
        mujoco.mj_step(model, data)

    for nm, qa in zip(obj_names, obj_qadr):
        bz = data.qpos[qa+2]
        print(f"  {nm} settled at z={bz:.4f} (expected ~{G1_TABLE_HEIGHT + BLOCK_HALF:.3f})")

    print(f"  Workspace: pick={pick_nm} at {desired_pick_xy.round(3)}, support={support_nm} at {desired_support_xy.round(3)}")

    # Support block indices for pinning (support stays stationary)
    if support_nm is not None:
        supp_idx = obj_names.index(support_nm)
        supp_qa = obj_qadr[supp_idx]
        supp_da = model.jnt_dofadr[obj_jids[supp_idx]]

    # Read actual settled block positions for IK targets (not hardcoded desired positions)
    pick_bid = obj_body_ids[obj_names.index(pick_nm)]
    pick_xy0 = data.xpos[pick_bid][:2].copy()

    if support_nm is not None:
        support_bid = obj_body_ids[obj_names.index(support_nm)]
        support_settled_pos = data.xpos[support_bid][:3].copy()
        place_xy0 = support_settled_pos[:2].copy()
    else:
        place_xy0 = pick_xy0 + np.array([0.12, -0.08])

    renderer = mujoco.Renderer(model, RENDER_H, RENDER_W)
    fd = OUT_DIR / f"_{task_name}_g1v10_frames"
    if fd.exists():
        shutil.rmtree(fd)
    fd.mkdir()

    finger_ctrl = FINGER_OPEN.copy()
    last_arm_q = SEED.copy()

    # Tracking variables for RET-03 diagnostics
    ik_failures = []     # list of (frame_idx, error_m) tuples
    n_clamped = 0        # count of frames where target was workspace-clamped
    palm_positions = []  # recorded palm positions for RMS error (RET-01)

    print(f"\nTrajectory: {n} frames, grip [{ls},{le}], pick={pick_nm}, support={support_nm}")

    # === RET-04: Trajectory workspace validation ===
    print(f"  Wrist X: [{wrist[:,0].min():.3f}, {wrist[:,0].max():.3f}]")
    print(f"  Wrist Y: [{wrist[:,1].min():.3f}, {wrist[:,1].max():.3f}]")
    print(f"  Wrist Z: [{wrist[:,2].min():.3f}, {wrist[:,2].max():.3f}]")
    print(f"  Workspace clamp: X[{WORKSPACE_MIN[0]:.2f},{WORKSPACE_MAX[0]:.2f}] "
          f"Y[{WORKSPACE_MIN[1]:.2f},{WORKSPACE_MAX[1]:.2f}] "
          f"Z[{WORKSPACE_MIN[2]:.2f},{WORKSPACE_MAX[2]:.2f}]")
    pick_in_x = wrist[:,0].min() <= desired_pick_xy[0] <= wrist[:,0].max()
    pick_in_y = wrist[:,1].min() <= desired_pick_xy[1] <= wrist[:,1].max()
    print(f"  Pick block ({desired_pick_xy.round(3)}) within wrist XY range: "
          f"X={'yes' if pick_in_x else 'NO'} Y={'yes' if pick_in_y else 'NO'}")

    for i in range(n):
        # === RET-01: Follow real wrist trajectory for ALL frames ===
        target = wrist[i].copy()
        target[2] = max(target[2], G1_TABLE_HEIGHT + 0.02)  # Z floor: 2cm above table

        # === RET-03: Clamp to reachable workspace envelope ===
        target_pre_clamp = target.copy()
        target = np.clip(target, WORKSPACE_MIN, WORKSPACE_MAX)
        if not np.allclose(target, target_pre_clamp, atol=1e-6):
            n_clamped += 1

        # === RET-02: Grasp intent from tracking data ===
        want_grip = bool(grasping[i] > 0)

        # IK solve
        for qa, v in zip(arm_qa, last_arm_q):
            data.qpos[qa] = v
        mujoco.mj_forward(model, data)
        arm_q = ik_solve(model, data, target, arm_qa, arm_da)
        last_arm_q = arm_q.copy()

        # --- RET-01/RET-03: Track palm position and check IK convergence ---
        pc = palm_center(model, data)
        palm_positions.append(pc.copy())
        ik_err = np.linalg.norm(target - pc)
        if ik_err > 0.02:  # 2cm convergence threshold
            ik_failures.append((i, float(ik_err)))

        # Finger control with distance-based pre-shaping (OUT-03)
        if want_grip:
            # Full closure for contact grasp — fast close to grip before block slips
            finger_target = FINGER_CLOSED
            blend_rate = 0.4
        else:
            # Distance-based pre-shaping: curl fingers as palm approaches block
            min_block_dist = min(
                np.linalg.norm(data.xpos[bid] - pc) for bid in obj_body_ids
            )
            preshape_t = np.clip(
                1.0 - (min_block_dist - PRESHAPE_DIST_FULL) / (PRESHAPE_DIST_START - PRESHAPE_DIST_FULL),
                0.0, 1.0
            )
            preshape_t = preshape_t * preshape_t  # quadratic ease-in
            if preshape_t > 0.01:
                finger_target = FINGER_OPEN + (FINGER_PRESHAPE - FINGER_OPEN) * preshape_t
                blend_rate = 0.15  # slower blend for anticipatory shaping
            else:
                finger_target = FINGER_OPEN
                blend_rate = 0.20
        finger_ctrl += (finger_target - finger_ctrl) * blend_rate
        data.ctrl[HAND_CTRL_START:HAND_CTRL_START+7] = finger_ctrl

        # Simulate substeps
        for _ in range(SUBSTEPS):
            freeze_robot()
            for qa, v in zip(arm_qa, arm_q):
                data.qpos[qa] = v
            for da in arm_da:
                data.qvel[da] = 0

            # Pin support block at its settled position
            if support_nm is not None:
                data.qpos[supp_qa:supp_qa + 3] = support_settled_pos
                data.qpos[supp_qa + 3:supp_qa + 7] = [1, 0, 0, 0]
                data.qvel[supp_da:supp_da + 6] = 0

            data.ctrl[HAND_CTRL_START:HAND_CTRL_START+7] = finger_ctrl
            mujoco.mj_step(model, data)

        renderer.update_scene(data, camera="front")
        Image.fromarray(renderer.render()).save(fd / f"frame_{i:04d}.png")

        if i % 10 == 0:
            pc = palm_center(model, data)
            block_zs = {nm: data.qpos[obj_qadr[obj_names.index(nm)]+2] for nm in obj_names}
            block_dists = {}
            for nm in obj_names:
                bi = obj_names.index(nm)
                bpos = data.qpos[obj_qadr[bi]:obj_qadr[bi]+3]
                block_dists[nm] = np.linalg.norm(pc - bpos)
            print(f"F{i:03d} grip={want_grip} palm={pc.round(3)} "
                  f"dists={{{', '.join(f'{k}:{v:.3f}' for k,v in block_dists.items())}}} "
                  f"block_z={{{', '.join(f'{k}:{v:.3f}' for k,v in block_zs.items())}}}")

    # === RET-01: Compute RMS tracking error ===
    palm_arr = np.array(palm_positions)
    target_arr = np.clip(wrist.copy(), WORKSPACE_MIN, WORKSPACE_MAX)
    target_arr[:, 2] = np.maximum(target_arr[:, 2], G1_TABLE_HEIGHT + 0.02)
    tracking_errors = np.linalg.norm(palm_arr - target_arr, axis=1)
    rms_err = float(np.sqrt(np.mean(tracking_errors**2)))
    mean_err = float(np.mean(tracking_errors))
    max_err = float(np.max(tracking_errors))
    print(f"\n  RET-01 Tracking: RMS={rms_err:.4f}m mean={mean_err:.4f}m max={max_err:.4f}m")
    if rms_err < 0.05:
        print(f"  RET-01 PASS: RMS error {rms_err:.4f}m < 0.05m threshold")
    else:
        print(f"  RET-01 WARNING: RMS error {rms_err:.4f}m exceeds 0.05m threshold")

    # === RET-03: Log clamping and convergence stats ===
    if n_clamped > 0:
        print(f"  RET-03 Clamping: {n_clamped}/{n} frames clamped to workspace envelope")
    else:
        print(f"  RET-03 Clamping: 0/{n} frames clamped (all targets within workspace)")
    if ik_failures:
        print(f"  RET-03 IK failures: {len(ik_failures)}/{n} frames with >2cm error")
        for frame, err in ik_failures[:5]:
            print(f"    Frame {frame:03d}: err={err:.4f}m")
        if len(ik_failures) > 5:
            print(f"    ... and {len(ik_failures)-5} more")
    else:
        print(f"  RET-03 IK failures: 0/{n} frames (all converged within 2cm)")

    for nm, qa in zip(obj_names, obj_qadr):
        pos = data.qpos[qa:qa+3]
        print(f"FINAL {nm}: xyz=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

    pos_a = data.qpos[obj_qadr[0]:obj_qadr[0]+3].copy()
    pos_b = data.qpos[obj_qadr[1]:obj_qadr[1]+3].copy() if len(obj_qadr) > 1 else np.zeros(3)
    za, zb = pos_a[2], pos_b[2]
    top_z = max(za, zb)
    bot_z = min(za, zb)
    z_ok = abs(top_z - bot_z - 2 * BLOCK_HALF) < 0.02
    xy_dist = np.linalg.norm(pos_a[:2] - pos_b[:2])
    xy_ok = xy_dist < 2 * BLOCK_HALF  # blocks must be horizontally aligned
    stacked = z_ok and xy_ok
    print(f"\n=== v10 RESULT ===")
    print(f"STACK CHECK: top_z={top_z:.3f} bot_z={bot_z:.3f} gap={top_z-bot_z:.3f} expected={2*BLOCK_HALF:.3f}")
    print(f"  Z aligned: {z_ok}  XY dist: {xy_dist:.3f}m (< {2*BLOCK_HALF:.3f}m): {xy_ok}")
    print(f"  STACKED={stacked}")
    print(f"pick={pick_nm} support={support_nm}")
    sep = np.linalg.norm(desired_pick_xy - desired_support_xy)
    print(f"Block separation: {sep:.3f}m")

    renderer.close()
    os.unlink(tmp.name)

    out = OUT_DIR / f"{task_name}_g1_v10.mp4"
    ff = shutil.which("ffmpeg")
    if not ff:
        raise RuntimeError("ffmpeg not found in PATH")
    subprocess.check_call([ff, "-y", "-framerate", str(FPS), "-i", str(fd/"frame_%04d.png"),
                           "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "fast", str(out)])
    print("OK", out)
    return out


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "stack2"
    render_task(task)
