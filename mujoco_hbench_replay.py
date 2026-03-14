#!/usr/bin/env python3
"""HumanoidBench H1+Shadow Hand: replay Flexa wrist trajectories.

Uses HumanoidBench's pre-built H1+Shadow Hand model and pre-trained reach
policy for whole-body control. Finger control is scripted based on grasp
detection from our trajectory data.

Architecture:
  1. Load h1hand_pos scene + stacking blocks
  2. Load pre-trained reach_one_hand policy (body control)
  3. For each frame: feed wrist target → reach policy → body joint commands
  4. Scripted finger close/open based on grasping flag
  5. MuJoCo steps physics, render frames → MP4

Output: pipeline/sim_renders/stack2_hbench.mp4
"""

import mujoco
import numpy as np
import torch
import json
import sys
import subprocess
import shutil
from pathlib import Path
from PIL import Image

# Paths
ASSETS = Path(__file__).parent / "humanoidbench_assets"
CALIB_DIR = Path(__file__).parent / "wrist_trajectories"
OUT_DIR = Path(__file__).parent / "sim_renders"
OUT_DIR.mkdir(exist_ok=True)

RENDER_W, RENDER_H = 640, 480
FPS = 10
BLOCK_HALF = 0.03  # 6cm blocks

# ── Reach policy loading ──────────────────────────────────────────────

class TorchModel(torch.nn.Module):
    def __init__(self, inputs, num_classes=1):
        super().__init__()
        self.dense1 = torch.nn.Linear(inputs, 256)
        self.dense2 = torch.nn.Linear(256, 256)
        self.dense3 = torch.nn.Linear(256, num_classes)

    def forward(self, x):
        x = torch.tanh(self.dense1(x))
        x = torch.tanh(self.dense2(x))
        x = self.dense3(x)
        return x


class ReachPolicy:
    def __init__(self, model_path, mean_path, var_path, input_dim=55, output_dim=19):
        self.model = TorchModel(input_dim, output_dim)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        self.model.eval()
        self.mean = np.load(mean_path)[0]
        self.var = np.load(var_path)[0]

    def __call__(self, obs):
        obs_norm = (obs - self.mean) / np.sqrt(self.var + 1e-8)
        with torch.no_grad():
            action = self.model(torch.from_numpy(obs_norm).float()).numpy()
        return np.clip(action, -1, 1)


# ── Model helpers ─────────────────────────────────────────────────────

def get_body_idxs(model):
    """Get body (non-hand) joint indices for reach policy observation.
    Matches HumanoidBench's get_body_idxs from wrappers.py."""
    body_idxs = []
    body_vel_idxs = []
    curr_idx = 0
    curr_vel_idx = 0
    for i in range(model.njnt):
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if jname and jname.startswith("free_"):
            if jname == "free_base":
                body_idxs.extend(list(range(curr_idx, curr_idx + 7)))
                body_vel_idxs.extend(list(range(curr_vel_idx, curr_vel_idx + 6)))
            curr_idx += 7
            curr_vel_idx += 6
            continue
        elif jname and not jname.startswith("lh_") and not jname.startswith("rh_") and "wrist" not in jname:
            body_idxs.append(curr_idx)
            body_vel_idxs.append(curr_vel_idx)
        curr_idx += 1
        curr_vel_idx += 1
    return body_idxs, body_vel_idxs


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


# ── Right hand finger grasp positions ────────────────────────────────
# These are "power grasp" joint targets for the right Shadow Hand
FINGER_OPEN = {}
FINGER_CLOSED = {
    "rh_A_THJ5": 0.8,   # thumb base rotation
    "rh_A_THJ4": 1.0,   # thumb proximal
    "rh_A_THJ3": 0.15,  # thumb hub
    "rh_A_THJ2": 0.5,   # thumb middle
    "rh_A_THJ1": 1.2,   # thumb distal
    "rh_A_FFJ4": 0.0,   # index knuckle
    "rh_A_FFJ3": 1.2,   # index proximal
    "rh_A_FFJ0": 2.5,   # index middle+distal (coupled)
    "rh_A_MFJ4": 0.0,   # middle knuckle
    "rh_A_MFJ3": 1.2,   # middle proximal
    "rh_A_MFJ0": 2.5,   # middle mid+distal
    "rh_A_RFJ4": 0.0,   # ring knuckle
    "rh_A_RFJ3": 1.2,   # ring proximal
    "rh_A_RFJ0": 2.5,   # ring mid+distal
    "rh_A_LFJ5": 0.5,   # little metacarpal
    "rh_A_LFJ4": 0.0,   # little knuckle
    "rh_A_LFJ3": 1.2,   # little proximal
    "rh_A_LFJ0": 2.5,   # little mid+distal
}


def build_stacking_scene():
    """Build the H1+Shadow Hand scene with a table and two blocks for stacking."""
    spec = mujoco.MjSpec.from_file(str(ASSETS / "envs" / "h1hand_pos_cube.xml"))

    # Add a table in front of the robot
    table = spec.worldbody.add_body()
    table.name = "table"
    table.pos = np.array([0.45, 0.0, 0.525])
    tg = table.add_geom()
    tg.type = mujoco.mjtGeom.mjGEOM_BOX
    tg.size = np.array([0.35, 0.50, 0.525])
    tg.rgba = np.array([0.55, 0.35, 0.18, 1.0])
    tg.mass = 100.0
    tg.friction = np.array([1.0, 0.005, 0.0001])
    tg.contype = 1
    tg.conaffinity = 1

    # Add a better camera for our stacking view
    cam = spec.worldbody.add_camera()
    cam.name = "front_view"
    cam.pos = np.array([1.8, 1.0, 1.4])
    cam.fovy = 50.0

    model = spec.compile()
    return model


def render_task(task_name: str):
    calib = json.loads((CALIB_DIR / f"{task_name}_calibrated.json").read_text())
    wrist = np.array(calib["wrist_sim"], dtype=float)
    grasping = np.array(calib["grasping"], dtype=float)
    n = len(wrist)

    # Load model
    model = build_stacking_scene()
    data = mujoco.MjData(model)

    # Reset to standing keyframe
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    print(f"Model: nq={model.nq} nv={model.nv} nu={model.nu} nbody={model.nbody}")

    # Get body joint indices (for reach policy obs)
    body_idxs, body_vel_idxs = get_body_idxs(model)
    print(f"Body idxs: {len(body_idxs)} joints, {len(body_vel_idxs)} vels")

    # Body actuator indices (the 19 that reach policy controls)
    # Indices: 0-14 (left side + torso) + 16-19 (right shoulder/elbow), skip 15 (left_wrist_yaw)
    body_act_idxs = list(range(15)) + list(range(16, 20))
    print(f"Body actuator idxs: {body_act_idxs}")

    # Right hand actuator mapping
    rh_act_map = {}
    for ai in range(model.nu):
        aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, ai) or ""
        if aname.startswith("rh_A_"):
            rh_act_map[aname] = ai
    print(f"Right hand actuators: {len(rh_act_map)}")

    # Load reach policy
    policy = ReachPolicy(
        str(ASSETS / "data" / "reach_one_hand" / "torch_model.pt"),
        str(ASSETS / "data" / "reach_one_hand" / "mean.npy"),
        str(ASSETS / "data" / "reach_one_hand" / "var.npy"),
        input_dim=55, output_dim=19,
    )
    print("Reach policy loaded (55→19)")

    # Find left_hand site for reach obs
    left_hand_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_hand")
    right_hand_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_hand")
    print(f"left_hand site: {left_hand_sid}, right_hand site: {right_hand_sid}")

    # Block body IDs
    block_names = ["left_cube_to_rotate", "right_cube_to_rotate"]
    block_bids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bn) for bn in block_names]

    # Reposition blocks for stacking task
    # Use right_cube as "pick" block near right hand, left_cube as "support"
    rh_palm_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rh_palm")
    mujoco.mj_forward(model, data)
    rh_pos = data.xpos[rh_palm_bid].copy()
    print(f"Right palm initial pos: {rh_pos.round(3)}")

    # Place pick block in front of right hand, support block offset
    # Find the freejoint qpos addresses for the cubes
    pick_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "free_right_cube_to_rotate")
    support_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "free_left_cube_to_rotate")
    pick_qa = model.jnt_qposadr[pick_jnt]
    support_qa = model.jnt_qposadr[support_jnt]

    # Table surface at z=1.05, place blocks on table near right hand
    table_z = 1.05 + BLOCK_HALF + 0.005
    pick_pos = np.array([rh_pos[0] + 0.08, rh_pos[1], table_z])
    support_pos = np.array([rh_pos[0] + 0.08, rh_pos[1] + 0.12, table_z])

    data.qpos[pick_qa:pick_qa+3] = pick_pos
    data.qpos[pick_qa+3:pick_qa+7] = [1, 0, 0, 0]
    data.qpos[support_qa:support_qa+3] = support_pos
    data.qpos[support_qa+3:support_qa+7] = [1, 0, 0, 0]

    # Save standing pose for freezing
    standing_qpos = data.qpos.copy()

    # Identify which joints to freeze (everything except right arm, right hand, and block freejoints)
    free_joints = set(["right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
                       "right_elbow", "right_wrist_yaw",
                       "free_right_cube_to_rotate", "free_left_cube_to_rotate"])
    # Add all right hand joints
    for j in range(model.njnt):
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if jname.startswith("rh_"):
            free_joints.add(jname)

    freeze_slices_q = []
    freeze_slices_v = []
    for j in range(model.njnt):
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"_j{j}"
        if jname in free_joints:
            continue
        qa = model.jnt_qposadr[j]
        da = model.jnt_dofadr[j]
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            freeze_slices_q.append((qa, 7))
            freeze_slices_v.append((da, 6))
        else:
            freeze_slices_q.append((qa, 1))
            freeze_slices_v.append((da, 1))

    def freeze_body():
        for qa, cnt in freeze_slices_q:
            data.qpos[qa:qa+cnt] = standing_qpos[qa:qa+cnt]
        for da, cnt in freeze_slices_v:
            data.qvel[da:da+cnt] = 0

    # Settle with frozen body
    for _ in range(500):
        freeze_body()
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    pick_settled = data.qpos[pick_qa:pick_qa+3].copy()
    support_settled = data.qpos[support_qa:support_qa+3].copy()
    print(f"Pick block settled: {pick_settled.round(3)}")
    print(f"Support block settled: {support_settled.round(3)}")

    # Grip window
    grip_idx = np.where(grasping > 0)[0]
    late = grip_idx[grip_idx >= int(0.6*n)]
    if len(late) < 3:
        late = grip_idx[-max(1, len(grip_idx)//2):] if len(grip_idx) else np.array([n-10])
    ls, le = int(late[0]), int(min(late[-1]+5, n-1))
    win = max(1, le - ls)
    print(f"Trajectory: {n} frames, grip window [{ls}, {le}]")

    # Grasp heights
    z_grasp = pick_settled[2] + 0.02
    z_hover = z_grasp + 0.12
    z_lift = z_grasp + 0.15

    # Action scaling (from HumanoidBench wrapper)
    action_low = np.array([model.actuator_ctrlrange[i][0] for i in body_act_idxs])
    action_high = np.array([model.actuator_ctrlrange[i][1] for i in body_act_idxs])

    def unnormalize_body_action(action):
        return (action + 1) / 2 * (action_high - action_low) + action_low

    def get_reach_obs(target):
        """Build observation for reach policy (matches SingleReachWrapper.get_reach_obs)."""
        position = data.qpos.flat.copy()[body_idxs]
        velocity = data.qvel.flat.copy()[body_vel_idxs]
        left_hand = data.site_xpos[left_hand_sid].copy()

        # Offset by pelvis XY (reach policy is trained with local coords)
        offset = np.array([position[0], position[1], 0])
        position_local = position.copy()
        position_local[:3] -= offset
        left_hand_local = left_hand - offset
        target_local = target - offset

        return np.concatenate((position_local[2:], velocity, left_hand_local, target_local))

    # Renderer
    renderer = mujoco.Renderer(model, RENDER_H, RENDER_W)
    fd = OUT_DIR / f"_{task_name}_hbench_frames"
    if fd.exists():
        shutil.rmtree(fd)
    fd.mkdir()

    # Note: reach policy controls LEFT hand. We need to adapt for RIGHT hand.
    # For now, let's use the reach policy as-is (left hand reaching) and see
    # if we can get basic trajectory following working. Then we can switch to
    # right hand or use the two-hand policy.
    #
    # Strategy: Use position control directly on right arm + scripted fingers.
    # The reach policy is trained for left hand only. For right hand, we'll
    # use the reach policy's body output (legs/torso for balance) and add
    # direct position targets for right arm.

    # Right arm actuator indices
    right_arm_acts = {
        "right_shoulder_pitch": 16,
        "right_shoulder_roll": 17,
        "right_shoulder_yaw": 18,
        "right_elbow": 19,
    }

    # Right arm joint qpos addresses
    right_arm_joints = {}
    for jname in ["right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow"]:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        right_arm_joints[jname] = model.jnt_qposadr[jid]

    # IK for right hand
    right_arm_qa = [right_arm_joints[k] for k in ["right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow"]]
    right_arm_da = []
    for jname in ["right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow"]:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        right_arm_da.append(model.jnt_dofadr[jid])

    # Also add right wrist yaw
    rw_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_wrist_yaw")
    if rw_jid >= 0:
        right_arm_qa.append(model.jnt_qposadr[rw_jid])
        right_arm_da.append(model.jnt_dofadr[rw_jid])

    def ik_right_hand(target, max_iter=80):
        """Simple IK to move right hand site to target."""
        jac = np.zeros((3, model.nv))
        jac_r = np.zeros((3, model.nv))
        for _ in range(max_iter):
            mujoco.mj_forward(model, data)
            hand_pos = data.site_xpos[right_hand_sid].copy()
            err = target - hand_pos
            if np.linalg.norm(err) < 0.005:
                break
            mujoco.mj_jacSite(model, data, jac, jac_r, right_hand_sid)
            J = jac[:, right_arm_da]
            JJT = J @ J.T + 0.005 * np.eye(3)
            dq = J.T @ np.linalg.solve(JJT, err)
            step = min(0.5, 0.2 + np.linalg.norm(err))
            for i, qa in enumerate(right_arm_qa):
                data.qpos[qa] += dq[i] * step

    # Main loop
    for i in range(n):
        p = 0.0 if (i < ls) else (1.0 if i > le else (i - ls) / win)

        # Compute target position for right hand
        support_cur = data.qpos[support_qa:support_qa+3].copy()
        place_xy = support_cur[:2]
        place_z = support_cur[2] + 2*BLOCK_HALF + 0.005

        if i < ls:
            # Pre-grip: hover over pick block
            target = np.array([pick_settled[0], pick_settled[1], z_hover])
            want_grip = False
        elif i > le:
            # Post-place: hover
            target = np.array([place_xy[0], place_xy[1], z_hover])
            want_grip = False
        else:
            # Grip sequence
            if p < 0.12:
                target = np.array([pick_settled[0], pick_settled[1], z_hover])
                want_grip = False
            elif p < 0.28:
                t = smoothstep((p - 0.12) / 0.16)
                z = z_hover + (z_grasp - z_hover) * t
                target = np.array([pick_settled[0], pick_settled[1], z])
                want_grip = t > 0.6
            elif p < 0.40:
                target = np.array([pick_settled[0], pick_settled[1], z_grasp])
                want_grip = True
            elif p < 0.55:
                t = smoothstep((p - 0.40) / 0.15)
                z = z_grasp + (z_lift - z_grasp) * t
                target = np.array([pick_settled[0], pick_settled[1], z])
                want_grip = True
            elif p < 0.75:
                t = smoothstep((p - 0.55) / 0.20)
                x = pick_settled[0] + (place_xy[0] - pick_settled[0]) * t
                y = pick_settled[1] + (place_xy[1] - pick_settled[1]) * t
                target = np.array([x, y, z_lift])
                want_grip = True
            elif p < 0.88:
                t = smoothstep((p - 0.75) / 0.13)
                z = z_lift + (place_z - z_lift) * t
                target = np.array([place_xy[0], place_xy[1], z])
                want_grip = True
            elif p < 0.94:
                target = np.array([place_xy[0], place_xy[1], place_z + 0.02])
                want_grip = False
            else:
                t = smoothstep((p - 0.94) / 0.06)
                z = place_z + 0.02 + (z_hover - place_z) * t
                target = np.array([place_xy[0], place_xy[1], z])
                want_grip = False

        # 1. IK for right hand to target
        ik_right_hand(target)

        # Set right arm actuators to match IK result
        for jname, ai in right_arm_acts.items():
            qa = right_arm_joints[jname]
            data.ctrl[ai] = data.qpos[qa]

        # Right wrist yaw
        rw_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_wrist_yaw")
        if rw_act >= 0 and rw_jid >= 0:
            data.ctrl[rw_act] = data.qpos[model.jnt_qposadr[rw_jid]]

        # 2. Finger control
        for aname, ai in rh_act_map.items():
            if want_grip and aname in FINGER_CLOSED:
                data.ctrl[ai] = FINGER_CLOSED[aname]
            else:
                data.ctrl[ai] = 0.0

        # 3. Step physics with frozen body
        substeps = max(10, min(50, int(1.0 / (FPS * model.opt.timestep))))
        for _ in range(substeps):
            freeze_body()
            # Also set right arm qpos from IK each substep
            for jname in right_arm_acts:
                qa = right_arm_joints[jname]
                data.qpos[qa] = data.qpos[qa]  # already set by IK
            mujoco.mj_step(model, data)

        # Render
        # Use cam_inhand or default camera
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_inhand")
        if cam_id < 0:
            cam_id = -1
        renderer.update_scene(data, camera=cam_id)
        Image.fromarray(renderer.render()).save(fd / f"frame_{i:04d}.png")

        # Log
        if i % 10 == 0:
            rh_pos = data.site_xpos[right_hand_sid]
            pick_pos_cur = data.qpos[pick_qa:pick_qa+3]
            sup_pos_cur = data.qpos[support_qa:support_qa+3]
            pelvis_z = data.qpos[2]
            dist_to_pick = np.linalg.norm(rh_pos - pick_pos_cur)
            print(f"F{i:03d} p={p:.2f} grip={want_grip} pelvis_z={pelvis_z:.3f} rh={rh_pos.round(3)} target={target.round(3)} dist_pick={dist_to_pick:.3f} pick_z={pick_pos_cur[2]:.3f} sup_z={sup_pos_cur[2]:.3f}")

    # Final positions
    pick_final = data.qpos[pick_qa:pick_qa+3]
    sup_final = data.qpos[support_qa:support_qa+3]
    print(f"\nFINAL pick: {pick_final.round(3)}")
    print(f"FINAL support: {sup_final.round(3)}")
    gap = abs(pick_final[2] - sup_final[2])
    stacked = gap > BLOCK_HALF and gap < 3*BLOCK_HALF and np.linalg.norm(pick_final[:2] - sup_final[:2]) < 0.05
    print(f"STACK CHECK: gap={gap:.3f} expected~{2*BLOCK_HALF:.3f} STACKED={stacked}")

    renderer.close()

    # Encode video
    out = OUT_DIR / f"{task_name}_hbench.mp4"
    ff = shutil.which("ffmpeg") or r"C:\Users\chris\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"
    subprocess.check_call([
        ff, "-y", "-framerate", str(FPS),
        "-i", str(fd / "frame_%04d.png"),
        "-vcodec", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "23", "-preset", "fast", str(out)
    ])
    print(f"OK {out}")
    return out


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "stack2"
    render_task(task)
