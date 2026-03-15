#!/usr/bin/env python3
"""G1 v9b: Enlarged colliders + adhesion actuators for real physics grasping.

No welds, no teleporting. Enlarged hand collision geoms make contact with blocks,
then MuJoCo adhesion actuators create physics-based sticking forces at contact points.
This models how a real gripper's friction/suction holds objects.

Output: pipeline/sim_renders/stack2_g1_v9b.mp4
"""

import mujoco
import numpy as np
import json, sys, tempfile, subprocess, shutil, os
from pathlib import Path
from PIL import Image

MENAGERIE = Path(r"C:\Users\chris\clawd\mujoco_menagerie\unitree_g1")
CALIB_DIR = Path(r"C:\Users\chris\clawd\pipeline\wrist_trajectories")
OUT_DIR = Path(r"C:\Users\chris\clawd\pipeline\sim_renders")
OUT_DIR.mkdir(exist_ok=True)

BLOCK_HALF = 0.03  # 6cm cubes (easier for hand to wrap around)
G1_TABLE_HEIGHT = 0.78  # raised: G1 arm can't reach low tables; keep near pelvis height for demo
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
FINGER_CLOSED = np.array([0.8, -1.2, -1.5, 1.5, 1.6, 1.5, 1.6])

HAND_COLLIDER_SCALE = 3.5
HAND_FRICTION = np.array([5.0, 0.05, 0.001])
FINGER_GAIN_MULTIPLIER = 25.0

# Bodies to attach adhesion actuators to
ADHESION_BODIES = [
    "right_hand_thumb_2_link",
    "right_hand_thumb_1_link",
    "right_hand_middle_0_link",
    "right_hand_middle_1_link", 
    "right_hand_index_0_link",
    "right_hand_index_1_link",
]
ADHESION_GAIN = 80.0  # force magnitude when ctrl=1


def scene_xml(obj_xml: str) -> str:
    # Build adhesion actuators for hand bodies
    adhesion_xml = "\n".join(
        f'    <adhesion body="{b}" ctrlrange="0 1" gain="{ADHESION_GAIN}"/>'
        for b in ADHESION_BODIES
    )
    
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

  <actuator>
{adhesion_xml}
  </actuator>
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
            f'<geom type="box" size="{BLOCK_HALF} {BLOCK_HALF} {BLOCK_HALF}" rgba="{c}" mass="0.05" '
            f'friction="2.0 0.01 0.001" contype="1" conaffinity="1"/>'
            f'</body>'
        )
    return "\n    ".join(parts)


def enlarge_hand_colliders(model):
    """Scale up all right hand collision geoms and increase friction."""
    hand_geom_ids = set()
    # Replace mesh collision geoms with scaled capsule primitives
    # Mesh geoms (type=7) don't respond to size scaling — their collision
    # shape comes from mesh vertices. We must change to primitive type.
    for gi in range(model.ngeom):
        bid = model.geom_bodyid[gi]
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if bname and "right_hand" in bname:
            if model.geom_contype[gi] > 0 or model.geom_conaffinity[gi] > 0:
                # Convert mesh geoms to capsules, scale box geoms
                if model.geom_type[gi] == mujoco.mjtGeom.mjGEOM_MESH:
                    # Replace mesh with a capsule approximation
                    model.geom_type[gi] = mujoco.mjtGeom.mjGEOM_CAPSULE
                    # Capsule size: [radius, half-length, 0]
                    # Use the mesh bounding box to estimate
                    bbox = model.geom_size[gi].copy()
                    radius = max(bbox[0], bbox[1]) * HAND_COLLIDER_SCALE
                    half_len = bbox[2] * HAND_COLLIDER_SCALE
                    model.geom_size[gi] = np.array([radius, half_len, 0.0])
                elif model.geom_type[gi] == mujoco.mjtGeom.mjGEOM_BOX:
                    model.geom_size[gi] *= HAND_COLLIDER_SCALE
                else:
                    model.geom_size[gi] *= HAND_COLLIDER_SCALE
                model.geom_friction[gi] = HAND_FRICTION
                hand_geom_ids.add(gi)
                print(f"    geom {gi} on {bname}: type={model.geom_type[gi]} size={model.geom_size[gi].round(4)}")
    
    # Increase finger actuator gains (only the original finger actuators, not adhesion)
    for ai in range(min(model.nu, HAND_CTRL_START + 7)):
        if ai >= HAND_CTRL_START:
            model.actuator_gainprm[ai, 0] *= FINGER_GAIN_MULTIPLIER
            model.actuator_biasprm[ai, 1] *= FINGER_GAIN_MULTIPLIER
    
    return hand_geom_ids


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

    # Use MjSpec to add contact pad geoms to finger bodies
    spec = mujoco.MjSpec.from_file(tmp.name)
    
    def find_body_spec(root, name):
        for b in root.bodies:
            if b.name == name:
                return b
            found = find_body_spec(b, name)
            if found:
                return found
        return None
    
    # Add modest sphere contact pads to fingertip bodies (physics-friendly contact)
    # Keep these small to avoid impulsive launches.
    pad_config = {
        "right_hand_thumb_2_link": 0.028,
        "right_hand_thumb_1_link": 0.022,
        "right_hand_middle_0_link": 0.022,
        "right_hand_middle_1_link": 0.028,
        "right_hand_index_0_link": 0.022,
        "right_hand_index_1_link": 0.028,
    }
    
    for bname, radius in pad_config.items():
        body = find_body_spec(spec.worldbody, bname)
        if body:
            g = body.add_geom()
            g.name = f"pad_{bname.replace('right_hand_', '')}"
            g.type = mujoco.mjtGeom.mjGEOM_SPHERE
            g.size = np.array([radius, 0.0, 0.0])
            g.contype = 1
            g.conaffinity = 1
            g.friction = np.array([5.0, 0.05, 0.001])
            g.mass = 0.001
            g.rgba = np.array([0.8, 0.8, 0.2, 0.3])  # semi-transparent yellow
            print(f"  Added contact pad to {bname}: sphere r={radius}")
    
    model = spec.compile()
    data = mujoco.MjData(model)

    print("Setting up hand colliders...")
    # Collect hand geom ids (including the new pads)
    hand_geom_ids = set()
    for gi in range(model.ngeom):
        bid = model.geom_bodyid[gi]
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if bname and ("right_hand" in bname or "right_wrist" in bname):
            if model.geom_contype[gi] > 0 or model.geom_conaffinity[gi] > 0:
                gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gi) or f"geom{gi}"
                print(f"    hand geom {gi}: {gname} on {bname} type={model.geom_type[gi]} size={model.geom_size[gi].round(4)}")
                hand_geom_ids.add(gi)
    
    # Increase finger actuator gains
    for ai in range(min(model.nu, HAND_CTRL_START + 7)):
        if ai >= HAND_CTRL_START:
            model.actuator_gainprm[ai, 0] *= FINGER_GAIN_MULTIPLIER
            model.actuator_biasprm[ai, 1] *= FINGER_GAIN_MULTIPLIER
    
    print(f"  {len(hand_geom_ids)} hand geoms (including pads)")

    # Find adhesion actuator indices
    adhesion_act_ids = []
    for b in ADHESION_BODIES:
        for ai in range(model.nu):
            aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, ai)
            # Adhesion actuators are named after the body
            if aname and aname == b:
                adhesion_act_ids.append(ai)
                break
        else:
            # Try matching by body transmission
            pass
    
    # If names don't match, find by index (adhesion actuators are after the original ones)
    if not adhesion_act_ids:
        # Original G1 has some actuators, our adhesion ones come after
        # Let's find them by checking actuator transmission type
        for ai in range(model.nu):
            if model.actuator_trntype[ai] == mujoco.mjtTrn.mjTRN_BODY:
                adhesion_act_ids.append(ai)
    
    print(f"  Found {len(adhesion_act_ids)} adhesion actuators at indices: {adhesion_act_ids}")

    key_id = jid(model, "stand", mujoco.mjtObj.mjOBJ_KEY)
    mujoco.mj_resetDataKeyframe(model, data, key_id)

    arm_qa, arm_da = [], []
    for jnm in RIGHT_ARM_JOINTS:
        j = jid(model, jnm, mujoco.mjtObj.mjOBJ_JOINT)
        arm_qa.append(model.jnt_qposadr[j])
        arm_da.append(model.jnt_dofadr[j])

    obj_jids = [jid(model, f"{nm}_jnt", mujoco.mjtObj.mjOBJ_JOINT) for nm in obj_names]
    obj_qadr = [model.jnt_qposadr[j] for j in obj_jids]

    for nm, qa in zip(obj_names, obj_qadr):
        xy = obj_xy[nm]
        data.qpos[qa:qa+3] = [xy[0] + 0.32 - 0.5, xy[1], G1_TABLE_HEIGHT + BLOCK_HALF + 0.01]
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

    # 2-phase settle
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
        data.qpos[qa:qa+3] = [xy[0] + 0.32 - 0.5, xy[1], G1_TABLE_HEIGHT + BLOCK_HALF + 0.01]
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

    # grip window
    n = len(wrist)
    grip_idx = np.where(grasping > 0)[0]
    late = grip_idx[grip_idx >= int(0.6*n)]
    if len(late) < 3:
        late = grip_idx[-max(1, len(grip_idx)//2):] if len(grip_idx) else np.array([n-10])
    ls, le = int(late[0]), int(min(late[-1]+5, n-1))
    win = max(1, le-ls)

    avg_xy = wrist[ls:le, :2].mean(axis=0)
    # Spread blocks apart for cleaner grasp (min 12cm between centers)
    if len(obj_names) >= 2:
        xy_a = obj_xy[obj_names[0]]
        xy_b = obj_xy[obj_names[1]]
        dist = np.linalg.norm(xy_a - xy_b)
        if dist < 0.12:
            center = (xy_a + xy_b) / 2
            direction = (xy_b - xy_a) / (dist + 1e-8)
            obj_xy[obj_names[0]] = center - direction * 0.06
            obj_xy[obj_names[1]] = center + direction * 0.06
            # Re-place objects with new positions
            for nm, qa in zip(obj_names, obj_qadr):
                xy = obj_xy[nm]
                data.qpos[qa] = xy[0] + 0.32 - 0.5
                data.qpos[qa+1] = xy[1]
            mujoco.mj_forward(model, data)
            print(f"  Blocks spread apart: {obj_names[0]}={obj_xy[obj_names[0]].round(4)}, {obj_names[1]}={obj_xy[obj_names[1]].round(4)}, dist={0.12:.3f}")

    pick_nm = min(obj_names, key=lambda nm: np.linalg.norm(obj_xy[nm] - avg_xy))
    support_nm = [x for x in obj_names if x != pick_nm][0] if len(obj_names) >= 2 else None

    # Workspace correction: place blocks explicitly in reachable workspace.
    # This is an INITIAL SCENE PLACEMENT adjustment (no mid-run teleports).
    table_x_offset = 0.32 - 0.5
    desired_pick_xy = np.array([0.38, -0.03])
    desired_support_xy = desired_pick_xy + np.array([-0.14, 0.10])  # keep support well away from wrist sweep

    # Update obj_xy (calib frame) so that sim placement hits these desired XYs
    obj_xy[pick_nm] = desired_pick_xy - np.array([table_x_offset, 0.0])
    if support_nm is not None:
        obj_xy[support_nm] = desired_support_xy - np.array([table_x_offset, 0.0])

    # Re-place objects with corrected positions
    for nm, qa in zip(obj_names, obj_qadr):
        xy = obj_xy[nm]
        data.qpos[qa:qa+3] = [xy[0] + table_x_offset, xy[1], G1_TABLE_HEIGHT + BLOCK_HALF + 0.01]
        data.qpos[qa+3:qa+7] = [1, 0, 0, 0]
        da = model.jnt_dofadr[obj_jids[obj_names.index(nm)]]
        data.qvel[da:da+6] = 0
    mujoco.mj_forward(model, data)
    print(f"  Workspace placement: pick={pick_nm} pick_xy={desired_pick_xy.round(3)} support={support_nm} support_xy={desired_support_xy.round(3)}")

    pick_xy0 = desired_pick_xy.copy()

    z_grasp = G1_TABLE_HEIGHT + BLOCK_HALF - 0.005  # slightly below block center so fingers wrap sides
    z_hover = z_grasp + 0.12
    z_lift = z_grasp + 0.15

    renderer = mujoco.Renderer(model, RENDER_H, RENDER_W)
    fd = OUT_DIR / f"_{task_name}_g1v9b_frames"
    if fd.exists():
        shutil.rmtree(fd)
    fd.mkdir()

    # Disable wrist collisions during pre-grip approach to prevent bumping blocks.
    wrist_geom_saved = {}
    for gi in range(model.ngeom):
        bid = model.geom_bodyid[gi]
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if bname and "right_wrist" in bname:
            wrist_geom_saved[gi] = (int(model.geom_contype[gi]), int(model.geom_conaffinity[gi]))
    
    finger_ctrl = FINGER_OPEN.copy()
    last_arm_q = SEED.copy()
    adhesion_on = False

    print(f"\nTrajectory: {n} frames, grip [{ls},{le}], pick={pick_nm}, support={support_nm}")

    for i in range(n):
        p = 0.0 if (i < ls) else (1.0 if i > le else (i-ls)/win)

        # Collision gating: keep wrist non-colliding during pre-grip approach only
        if i < ls:
            for gi in wrist_geom_saved:
                model.geom_contype[gi] = 0
                model.geom_conaffinity[gi] = 0
        else:
            for gi, (ct, ca) in wrist_geom_saved.items():
                model.geom_contype[gi] = ct
                model.geom_conaffinity[gi] = ca

        if support_nm is not None:
            sidx = obj_names.index(support_nm)
            support_pos = data.qpos[obj_qadr[sidx]:obj_qadr[sidx]+3].copy()
            place_xy = support_pos[:2]
            place_z = support_pos[2] + 2*BLOCK_HALF + 0.002
        else:
            place_xy = pick_xy0 + np.array([0.12, -0.08])
            place_z = G1_TABLE_HEIGHT + BLOCK_HALF

        if i < ls:
            # Pre-grip: go straight to hover over pick (avoid sweeping through blocks)
            target = np.array([pick_xy0[0], pick_xy0[1], z_hover])
            want_grip = False
        elif i > le:
            # Post-grip: retreat to hover over place
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
            elif p < 0.94:
                target = np.array([place_xy[0], place_xy[1], place_z + 0.02])
                want_grip = False
            else:
                t = smoothstep((p-0.94)/0.06)
                target = np.array([place_xy[0], place_xy[1], place_z + 0.02 + (z_hover-place_z)*t])
                want_grip = False

        # IK
        for qa, v in zip(arm_qa, last_arm_q):
            data.qpos[qa] = v
        mujoco.mj_forward(model, data)
        arm_q = ik_solve(model, data, target, arm_qa, arm_da)
        last_arm_q = arm_q.copy()

        # finger ctrl
        target_ctrl = FINGER_CLOSED if want_grip else FINGER_OPEN
        finger_ctrl += (target_ctrl - finger_ctrl) * 0.25
        data.ctrl[HAND_CTRL_START:HAND_CTRL_START+7] = finger_ctrl

        # adhesion control
        adhesion_val = 1.0 if want_grip else 0.0
        for aid in adhesion_act_ids:
            data.ctrl[aid] = adhesion_val

        # simulate
        for _ in range(SUBSTEPS):
            freeze_robot()
            for qa, v in zip(arm_qa, arm_q):
                data.qpos[qa] = v
            for da in arm_da:
                data.qvel[da] = 0
            data.ctrl[HAND_CTRL_START:HAND_CTRL_START+7] = finger_ctrl
            for aid in adhesion_act_ids:
                data.ctrl[aid] = adhesion_val
            mujoco.mj_step(model, data)

        renderer.update_scene(data, camera="front")
        Image.fromarray(renderer.render()).save(fd / f"frame_{i:04d}.png")

        if i % 10 == 0:
            pc = palm_center(model, data)
            contacts = 0
            for c in range(data.ncon):
                g1, g2 = data.contact[c].geom1, data.contact[c].geom2
                if (g1 in hand_geom_ids and g2 in block_geom_ids) or \
                   (g1 in block_geom_ids and g2 in hand_geom_ids):
                    contacts += 1
                # Debug: print any contact involving blocks during grip
                if want_grip and (g1 in block_geom_ids or g2 in block_geom_ids):
                    n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1) or f"g{g1}"
                    n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2) or f"g{g2}"
                    b1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g1]) or "?"
                    b2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g2]) or "?"
                    print(f"  CONTACT: {n1}({b1}) <-> {n2}({b2}) dist={data.contact[c].dist:.4f}")
            block_zs = {nm: data.qpos[obj_qadr[obj_names.index(nm)]+2] for nm in obj_names}
            pc = palm_center(model, data)
            # Distance from palm to each block
            block_dists = {}
            for nm in obj_names:
                bi = obj_names.index(nm)
                bpos = data.qpos[obj_qadr[bi]:obj_qadr[bi]+3]
                block_dists[nm] = np.linalg.norm(pc - bpos)
            if want_grip and i % 10 == 0:
                for nm in obj_names:
                    bi = obj_names.index(nm)
                    bpos = data.qpos[obj_qadr[bi]:obj_qadr[bi]+3]
                    print(f"  {nm} pos={bpos.round(4)}")
            print(f"F{i:03d} p={p:.2f} grip={want_grip} adh={adhesion_val:.0f} contacts={contacts} palm={pc.round(3)} dists={{{', '.join(f'{k}:{v:.3f}' for k,v in block_dists.items())}}} block_z={{{', '.join(f'{k}:{v:.3f}' for k,v in block_zs.items())}}}")

    for nm, qa in zip(obj_names, obj_qadr):
        pos = data.qpos[qa:qa+3]
        print(f"FINAL {nm}: xyz=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

    za = data.qpos[obj_qadr[0]+2]
    zb = data.qpos[obj_qadr[1]+2] if len(obj_qadr) > 1 else 0
    top_z = max(za, zb)
    bot_z = min(za, zb)
    stacked = abs(top_z - bot_z - 2*BLOCK_HALF) < 0.02
    print(f"STACK CHECK: top_z={top_z:.3f} bot_z={bot_z:.3f} gap={top_z-bot_z:.3f} expected={2*BLOCK_HALF:.3f} STACKED={stacked}")

    renderer.close()
    os.unlink(tmp.name)

    out = OUT_DIR / f"{task_name}_g1_v9b.mp4"
    ff = shutil.which("ffmpeg") or r"C:\Users\chris\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"
    subprocess.check_call([ff, "-y", "-framerate", str(FPS), "-i", str(fd/"frame_%04d.png"),
                           "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "fast", str(out)])
    print("OK", out)
    return out


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "stack2"
    render_task(task)
