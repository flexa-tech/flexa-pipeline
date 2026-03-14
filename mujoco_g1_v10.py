#!/usr/bin/env python3
"""G1 v10: Kinematic block attachment (Franka v9 pattern).

Rewritten to use the same kinematic-attach approach as mujoco_franka_v9.py:
- Arm joints set kinematically via IK
- Block grasping uses smooth-blend kinematic attachment (no adhesion)
- Fingers close visually during grip
- No block pinning, no adhesion actuators, no contact pads

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
SUBSTEPS = max(10, min(50, int(1.0 / (FPS * TIMESTEP))))

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
FINGER_CLOSED = np.array([0.4, -0.5, -0.6, 0.8, 0.9, 0.8, 0.9])  # partial closure — wrap around block, not through it

FINGER_GAIN_MULTIPLIER = 25.0
BLEND_FRAMES = 12  # smooth blend from rest to hand-tracking


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
            f'friction="2.0 0.01 0.001" contype="1" conaffinity="1"/>'
            f'</body>'
        )
    return "\n    ".join(parts)


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def render_task(task_name: str):
    calib = json.loads((CALIB_DIR / f"{task_name}_calibrated.json").read_text())
    wrist = np.array(calib["wrist_sim"], dtype=float)
    grasping = np.array(calib["grasping"], dtype=float)
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

    # Workspace correction: place blocks in reachable workspace
    table_x_offset = 0.32 - 0.5
    desired_pick_xy = np.array([0.38, -0.03])
    desired_support_xy = desired_pick_xy + np.array([0.0, -0.28])

    # Determine pick vs support
    n = len(wrist)
    grip_idx = np.where(grasping > 0)[0]
    late = grip_idx[grip_idx >= int(0.6*n)]
    if len(late) < 3:
        late = grip_idx[-max(1, len(grip_idx)//2):] if len(grip_idx) else np.array([n-10])
    ls, le = int(late[0]), int(min(late[-1]+5, n-1))
    win = max(1, le-ls)

    avg_xy = wrist[ls:le, :2].mean(axis=0)
    pick_nm = min(obj_names, key=lambda nm: np.linalg.norm(obj_xy[nm] - avg_xy))
    support_nm = [x for x in obj_names if x != pick_nm][0] if len(obj_names) >= 2 else None

    # Update positions to desired workspace
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

    # Trajectory heights
    z_grasp = G1_TABLE_HEIGHT + BLOCK_HALF - 0.005
    z_hover = z_grasp + 0.12
    z_lift = z_grasp + 0.18

    renderer = mujoco.Renderer(model, RENDER_H, RENDER_W)
    fd = OUT_DIR / f"_{task_name}_g1v10_frames"
    if fd.exists():
        shutil.rmtree(fd)
    fd.mkdir()

    finger_ctrl = FINGER_OPEN.copy()
    last_arm_q = SEED.copy()

    # --- Kinematic block attachment state (Franka v9 pattern) ---
    grasped_obj = None       # which object name is currently grasped
    grasped_bid = None       # body id of grasped object
    grasped_jnt_adr = None   # qpos address of grasped object's freejoint
    grasp_offset = None      # relative XYZ offset (block_pos - palm_pos at grasp time)
    grasp_start_pos = None   # block position at grasp time (for smooth blend)
    grasp_quat = None        # block orientation at grasp time
    grasp_age = 0            # frames since grasp started
    # Post-release: brief settle at actual position (prevents bounce, then clears)
    placed_jnt_adr = None
    placed_dof_adr = None
    placed_pos = None
    placed_countdown = 0
    SETTLE_FRAMES = 50  # pin through hand retreat to prevent finger-block collision bounce

    print(f"\nTrajectory: {n} frames, grip [{ls},{le}], pick={pick_nm}, support={support_nm}")

    for i in range(n):
        p = 0.0 if (i < ls) else (1.0 if i > le else (i-ls)/win)

        # Place target — use actual block positions from sim
        palm_z_offset = -grasp_offset[2] if grasp_offset is not None else 0.02
        if support_nm is not None:
            place_xy = place_xy0
            block_stack_z = G1_TABLE_HEIGHT + 3 * BLOCK_HALF + 0.002
            place_z = block_stack_z + palm_z_offset
        else:
            place_xy = place_xy0
            place_z = G1_TABLE_HEIGHT + BLOCK_HALF + palm_z_offset

        # Compute IK target and grip intent
        if i < ls:
            # Use real wrist trajectory for natural pre-grasp arm motion
            wrist_pos = wrist[i]
            target = np.array([wrist_pos[0], wrist_pos[1],
                               max(wrist_pos[2], G1_TABLE_HEIGHT + 0.15)])
            want_grip = False
        elif i > le:
            target = np.array([place_xy[0], place_xy[1], z_hover])
            want_grip = False
        else:
            if p < 0.12:
                target = np.array([pick_xy0[0], pick_xy0[1], z_hover])
                want_grip = False
            elif p < 0.25:
                t = smoothstep((p-0.12)/0.13)
                target = np.array([pick_xy0[0], pick_xy0[1], z_hover + (z_grasp - z_hover)*t])
                want_grip = t > 0.5
            elif p < 0.42:
                # Dwell at grasp height — fingers closing, attachment triggers
                target = np.array([pick_xy0[0], pick_xy0[1], z_grasp])
                want_grip = True
            elif p < 0.55:
                t = smoothstep((p-0.42)/0.13)
                target = np.array([pick_xy0[0], pick_xy0[1], z_grasp + (z_lift - z_grasp)*t])
                want_grip = True
            elif p < 0.75:
                t = smoothstep((p-0.55)/0.20)
                mid_xy = (pick_xy0 + place_xy) / 2 + np.array([0.0, -0.08])
                if t < 0.5:
                    t2 = t * 2
                    cx = pick_xy0[0] + (mid_xy[0]-pick_xy0[0])*t2
                    cy = pick_xy0[1] + (mid_xy[1]-pick_xy0[1])*t2
                else:
                    t2 = (t-0.5)*2
                    cx = mid_xy[0] + (place_xy[0]-mid_xy[0])*t2
                    cy = mid_xy[1] + (place_xy[1]-mid_xy[1])*t2
                target = np.array([cx, cy, z_lift])
                want_grip = True
            elif p < 0.88:
                t = smoothstep((p-0.75)/0.13)
                target = np.array([place_xy[0], place_xy[1], z_lift + (place_z-z_lift)*t])
                want_grip = True
            elif p < 0.92:
                # Hold at stack position — block settles onto support
                target = np.array([place_xy[0], place_xy[1], place_z])
                want_grip = True
            elif p < 0.95:
                # Release and retreat slightly up
                target = np.array([place_xy[0], place_xy[1], place_z + 0.04])
                want_grip = False
            else:
                t = smoothstep((p-0.95)/0.05)
                target = np.array([place_xy[0], place_xy[1], place_z + 0.04 + (z_hover-place_z)*t])
                want_grip = False

        # IK solve
        for qa, v in zip(arm_qa, last_arm_q):
            data.qpos[qa] = v
        mujoco.mj_forward(model, data)
        arm_q = ik_solve(model, data, target, arm_qa, arm_da)
        last_arm_q = arm_q.copy()

        # --- Kinematic block attachment logic (Franka v9 pattern) ---
        pc = palm_center(model, data)

        # ATTACH: when want_grip=True and palm is close enough to a block
        if want_grip and grasped_obj is None:
            best_dist, best_name, best_bid2 = 1e9, None, None
            for name, bid in zip(obj_names, obj_body_ids):
                bpos = data.xpos[bid].copy()
                xy_dist = np.linalg.norm(bpos[:2] - pc[:2])
                z_gap = abs(pc[2] - bpos[2])
                if xy_dist < best_dist:
                    best_dist = xy_dist
                    best_name = name
                    best_bid2 = bid
            if best_dist < 0.15 and abs(pc[2] - data.xpos[best_bid2][2]) < 0.08:
                grasped_obj = best_name
                grasped_bid = best_bid2
                jnt_name = f"{best_name}_jnt"
                jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
                grasped_jnt_adr = model.jnt_qposadr[jnt_id]
                block_pos = data.xpos[grasped_bid].copy()
                grasp_offset = block_pos - pc
                grasp_start_pos = block_pos.copy()
                grasp_quat = data.qpos[grasped_jnt_adr+3:grasped_jnt_adr+7].copy()
                grasp_age = 0
                print(f"  F{i:03d} GRASP: {best_name} (dist={best_dist:.3f})")

        # RELEASE: brief settle at actual XY + correct stack Z (no teleport)
        if not want_grip and grasped_obj is not None:
            jnt_idx = obj_jids[obj_names.index(grasped_obj)]
            dof_adr = model.jnt_dofadr[jnt_idx]
            data.qvel[dof_adr:dof_adr + 6] = 0
            # Pin at actual XY (where hand carried it) + correct stack Z
            if support_nm is not None:
                stack_z = support_settled_pos[2] + 2 * BLOCK_HALF
                placed_jnt_adr = grasped_jnt_adr
                placed_dof_adr = dof_adr
                placed_pos = np.array([
                    data.qpos[grasped_jnt_adr],      # actual X from carry
                    data.qpos[grasped_jnt_adr + 1],   # actual Y from carry
                    stack_z                             # exact contact stack Z
                ])
                placed_countdown = SETTLE_FRAMES
            print(f"  F{i:03d} RELEASE: {grasped_obj} at z={data.qpos[grasped_jnt_adr+2]:.3f}")
            grasped_obj = None
            grasped_bid = None
            grasped_jnt_adr = None
            grasp_offset = None
            grasp_start_pos = None
            grasp_quat = None
            grasp_age = 0

        # Increment blend counter
        if grasped_obj is not None:
            grasp_age += 1

        # Finger control
        target_ctrl = FINGER_CLOSED if want_grip else FINGER_OPEN
        finger_ctrl += (target_ctrl - finger_ctrl) * 0.25
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

            # Post-release: brief settle pin (clears after SETTLE_FRAMES)
            if placed_jnt_adr is not None and placed_countdown > 0:
                data.qpos[placed_jnt_adr:placed_jnt_adr+3] = placed_pos
                data.qpos[placed_jnt_adr+3:placed_jnt_adr+7] = [1, 0, 0, 0]
                data.qvel[placed_dof_adr:placed_dof_adr+6] = 0

            # Kinematic block attachment: block tracks palm with smooth onset
            if grasped_jnt_adr is not None:
                pc_sub = palm_center(model, data)
                target_pos = pc_sub + grasp_offset
                # Smooth blend over BLEND_FRAMES
                blend_t = min(1.0, grasp_age / BLEND_FRAMES)
                blend_t = blend_t * blend_t * (3 - 2 * blend_t)  # smoothstep
                blended_pos = np.array([
                    grasp_start_pos[0] * (1 - blend_t) + target_pos[0] * blend_t,
                    grasp_start_pos[1] * (1 - blend_t) + target_pos[1] * blend_t,
                    max(grasp_start_pos[2], target_pos[2])  # Z always goes UP
                ])
                data.qpos[grasped_jnt_adr:grasped_jnt_adr+3] = blended_pos
                data.qpos[grasped_jnt_adr+3:grasped_jnt_adr+7] = grasp_quat
                # Zero velocity so physics doesn't fight the kinematic hold
                jnt_idx = obj_jids[obj_names.index(grasped_obj)]
                dof_adr = model.jnt_dofadr[jnt_idx]
                data.qvel[dof_adr:dof_adr+6] = 0

            data.ctrl[HAND_CTRL_START:HAND_CTRL_START+7] = finger_ctrl
            mujoco.mj_step(model, data)

        # Decrement settle countdown; clear state when done
        if placed_countdown > 0:
            placed_countdown -= 1
            if placed_countdown == 0:
                placed_jnt_adr = None
                placed_dof_adr = None
                placed_pos = None

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
            attached = grasped_obj or "none"
            print(f"F{i:03d} p={p:.2f} grip={want_grip} attached={attached} palm={pc.round(3)} "
                  f"dists={{{', '.join(f'{k}:{v:.3f}' for k,v in block_dists.items())}}} "
                  f"block_z={{{', '.join(f'{k}:{v:.3f}' for k,v in block_zs.items())}}}")

    for nm, qa in zip(obj_names, obj_qadr):
        pos = data.qpos[qa:qa+3]
        print(f"FINAL {nm}: xyz=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

    za = data.qpos[obj_qadr[0]+2]
    zb = data.qpos[obj_qadr[1]+2] if len(obj_qadr) > 1 else 0
    top_z = max(za, zb)
    bot_z = min(za, zb)
    stacked = abs(top_z - bot_z - 2*BLOCK_HALF) < 0.02
    print(f"\n=== v10 RESULT ===")
    print(f"STACK CHECK: top_z={top_z:.3f} bot_z={bot_z:.3f} gap={top_z-bot_z:.3f} expected={2*BLOCK_HALF:.3f} STACKED={stacked}")
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
