#!/usr/bin/env python3
"""Bimanual side-by-side: R3D capture (left) vs H1 with both Shadow Hands (right).

Uses:
- retargeted/stack2_right_shadow_hand.json (right hand finger joints)
- retargeted/stack2_left_shadow_hand.json (left hand finger joints)
- r3d_output/stack2_rgb/ (original capture frames)
- HumanoidBench H1+Shadow Hand model
- Book-shaped objects on table

Both arms driven by wrist trajectory, both Shadow Hands by dex-retargeted joints.
"""
import mujoco, numpy as np, json, sys
from pathlib import Path
from PIL import Image, ImageDraw
import shutil, subprocess

ASSETS = Path(__file__).parent / "humanoidbench_assets"
RETARGET = Path(__file__).parent / "retargeted"
RGB_DIR = Path(__file__).parent / "r3d_output" / "stack2_rgb"
CALIB = Path(__file__).parent / "wrist_trajectories"
OUT = Path(__file__).parent / "sim_renders"
OUT.mkdir(exist_ok=True)

W, H = 480, 360


def render():
    # Load retargeted data
    ret_r = json.loads((RETARGET / "stack2_right_shadow_hand.json").read_text())
    ret_l = json.loads((RETARGET / "stack2_left_shadow_hand.json").read_text())
    jnames_r = ret_r["joint_names"]
    jnames_l = ret_l["joint_names"]
    n_ret = len(ret_r["results"])
    print(f"Retargeted: {n_ret} frames per hand")

    # Load wrist trajectory for arm motion
    calib = json.loads((CALIB / "stack2_calibrated.json").read_text())
    wrist = np.array(calib["wrist_sim"], dtype=float)
    grasping = np.array(calib["grasping"], dtype=float)
    n_traj = len(wrist)

    # RGB frames
    rgb_files = sorted(RGB_DIR.glob("*.jpg"))
    n_rgb = len(rgb_files)
    print(f"RGB: {n_rgb}, Trajectory: {n_traj}, Retargeted: {n_ret}")

    # Build scene
    spec = mujoco.MjSpec.from_file(str(ASSETS / "envs" / "h1hand_pos_cube.xml"))

    # Add table
    table = spec.worldbody.add_body()
    table.name = "table"
    table.pos = np.array([0.45, 0.0, 0.475])
    tg = table.add_geom()
    tg.type = mujoco.mjtGeom.mjGEOM_BOX
    tg.size = np.array([0.30, 0.40, 0.475])
    tg.rgba = np.array([0.55, 0.35, 0.18, 1.0])
    tg.mass = 100.0
    tg.contype = 1
    tg.conaffinity = 1

    # Replace cubes with book-shaped objects
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)

    # Resize cube geoms to book shape (25cm x 18cm x 3cm)
    for name in ["left_cube_to_rotate", "right_cube_to_rotate"]:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        for gi in range(m.ngeom):
            if m.geom_bodyid[gi] == bid:
                m.geom_size[gi] = [0.125, 0.09, 0.015]  # book half-sizes

    mujoco.mj_forward(m, d)

    # Place books on table after settle
    pick_jnt = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "free_right_cube_to_rotate")
    sup_jnt = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "free_left_cube_to_rotate")
    pick_qa = m.jnt_qposadr[pick_jnt]
    sup_qa = m.jnt_qposadr[sup_jnt]
    table_top = 0.95
    book_z = table_top + 0.015 + 0.001

    # Build actuator maps for both hands
    rh_act_map, lh_act_map = {}, {}
    for ai in range(m.nu):
        an = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, ai) or ""
        if an.startswith("rh_A_"):
            rh_act_map[an[5:]] = ai  # strip "rh_A_"
        elif an.startswith("lh_A_"):
            lh_act_map[an[5:]] = ai
    print(f"Right actuators: {len(rh_act_map)}, Left actuators: {len(lh_act_map)}")

    # Arm joints for both sides
    arm_names_r = ["right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist_yaw"]
    arm_names_l = ["left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist_yaw"]

    arm_qa_r = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)] for jn in arm_names_r]
    arm_qa_l = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)] for jn in arm_names_l]
    arm_da_r = [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)] for jn in arm_names_r]
    arm_da_l = [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)] for jn in arm_names_l]

    rh_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_hand")
    lh_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "left_hand")

    arm_jids_r = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn) for jn in arm_names_r]
    arm_jids_l = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn) for jn in arm_names_l]
    arm_lo_r = np.array([m.jnt_range[j][0] for j in arm_jids_r])
    arm_hi_r = np.array([m.jnt_range[j][1] for j in arm_jids_r])
    arm_lo_l = np.array([m.jnt_range[j][0] for j in arm_jids_l])
    arm_hi_l = np.array([m.jnt_range[j][1] for j in arm_jids_l])

    arm_acts_r, arm_acts_l = {}, {}
    for ai in range(m.nu):
        an = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, ai) or ""
        if an in arm_names_r: arm_acts_r[an] = ai
        if an in arm_names_l: arm_acts_l[an] = ai

    # Freeze body
    standing = d.qpos.copy()
    free_set = set(arm_names_r + arm_names_l)
    for j in range(m.njnt):
        jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if jn.startswith("rh_") or jn.startswith("lh_") or "cube" in jn:
            free_set.add(jn)

    freeze_q, freeze_v = [], []
    for j in range(m.njnt):
        jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or f"_unnamed_{j}"
        if jn in free_set: continue
        qa, da = m.jnt_qposadr[j], m.jnt_dofadr[j]
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            freeze_q.append((qa, 7)); freeze_v.append((da, 6))
        else:
            freeze_q.append((qa, 1)); freeze_v.append((da, 1))

    def freeze():
        for qa, cnt in freeze_q:
            d.qpos[qa:qa+cnt] = standing[qa:qa+cnt]
        for da, cnt in freeze_v:
            d.qvel[da:da+cnt] = 0

    # Settle
    for _ in range(200):
        freeze()
        mujoco.mj_step(m, d)

    # Place books on table
    d.qpos[pick_qa:pick_qa+3] = [0.40, -0.10, book_z]
    d.qpos[pick_qa+3:pick_qa+7] = [1, 0, 0, 0]
    d.qpos[sup_qa:sup_qa+3] = [0.40, 0.10, book_z]
    d.qpos[sup_qa+3:sup_qa+7] = [1, 0, 0, 0]
    d.qvel[m.jnt_dofadr[pick_jnt]:m.jnt_dofadr[pick_jnt]+6] = 0
    d.qvel[m.jnt_dofadr[sup_jnt]:m.jnt_dofadr[sup_jnt]+6] = 0
    mujoco.mj_forward(m, d)

    # Wrist trajectory mapping — use relative offsets centered on default hand positions
    rh_default = d.site_xpos[rh_sid].copy()
    lh_default = d.site_xpos[lh_sid].copy()
    print(f"RH default: {rh_default.round(3)}, LH default: {lh_default.round(3)}")

    ws = wrist.copy()
    ws_mean = ws.mean(axis=0)
    ws_centered = ws - ws_mean
    ws_max_abs = np.max(np.abs(ws_centered), axis=0)
    ws_max_abs[ws_max_abs < 0.01] = 1.0
    arm_reach = 0.15
    scale = arm_reach / ws_max_abs

    def traj_to_target_r(idx):
        offset = ws_centered[idx % n_traj] * scale
        return rh_default + offset

    def traj_to_target_l(idx):
        # Mirror the trajectory for left hand (flip Y)
        offset = ws_centered[idx % n_traj] * scale
        offset[1] = -offset[1]  # mirror Y for left side
        return lh_default + offset

    # IK function
    jac = np.zeros((3, m.nv))
    jac_r = np.zeros((3, m.nv))

    def ik_arm(target, site_id, arm_qa, arm_da, arm_lo, arm_hi, max_iter=60):
        for _ in range(max_iter):
            mujoco.mj_forward(m, d)
            pos = d.site_xpos[site_id].copy()
            err = target - pos
            if np.linalg.norm(err) < 0.005: break
            mujoco.mj_jacSite(m, d, jac, jac_r, site_id)
            J = jac[:, arm_da]
            JJT = J @ J.T + 0.005 * np.eye(3)
            dq = J.T @ np.linalg.solve(JJT, err)
            for i, qa in enumerate(arm_qa):
                d.qpos[qa] = np.clip(d.qpos[qa] + dq[i] * 0.4, arm_lo[i], arm_hi[i])

    # Render
    renderer = mujoco.Renderer(m, H, W)
    fd = OUT / "_bimanual_sidebyside_frames"
    if fd.exists(): shutil.rmtree(fd)
    fd.mkdir()

    n_frames = min(n_traj, n_ret // 2)  # use trajectory frame count
    print(f"Rendering {n_frames} frames...")

    for i in range(n_frames):
        # Map to retargeted frame index (2:1 ratio since R3D is 30fps, traj is subsampled)
        ret_idx = min(i * 2, n_ret - 1)
        rgb_idx = min(int(i * n_rgb / n_frames), n_rgb - 1)

        # IK both arms
        target_r = traj_to_target_r(i)
        target_l = traj_to_target_l(i)
        ik_arm(target_r, rh_sid, arm_qa_r, arm_da_r, arm_lo_r, arm_hi_r)
        ik_arm(target_l, lh_sid, arm_qa_l, arm_da_l, arm_lo_l, arm_hi_l)

        # Set arm actuators
        for an, ai in arm_acts_r.items():
            idx = arm_names_r.index(an)
            d.ctrl[ai] = d.qpos[arm_qa_r[idx]]
        for an, ai in arm_acts_l.items():
            idx = arm_names_l.index(an)
            d.ctrl[ai] = d.qpos[arm_qa_l[idx]]

        # Apply retargeted finger joints — RIGHT hand
        ret_frame_r = ret_r["results"][ret_idx]
        if ret_frame_r["joints"] is not None:
            for ji, jn in enumerate(jnames_r):
                if jn.startswith("dummy_"): continue
                if jn in rh_act_map:
                    ai = rh_act_map[jn]
                    d.ctrl[ai] = np.clip(ret_frame_r["joints"][ji],
                                         m.actuator_ctrlrange[ai][0],
                                         m.actuator_ctrlrange[ai][1])

        # Apply retargeted finger joints — LEFT hand
        ret_frame_l = ret_l["results"][ret_idx]
        if ret_frame_l["joints"] is not None:
            for ji, jn in enumerate(jnames_l):
                if jn.startswith("dummy_"): continue
                if jn in lh_act_map:
                    ai = lh_act_map[jn]
                    d.ctrl[ai] = np.clip(ret_frame_l["joints"][ji],
                                         m.actuator_ctrlrange[ai][0],
                                         m.actuator_ctrlrange[ai][1])

        # Step
        for _ in range(20):
            freeze()
            mujoco.mj_step(m, d)

        mujoco.mj_forward(m, d)

        # Render robot
        cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "cam_inhand")
        renderer.update_scene(d, camera=cam_id if cam_id >= 0 else -1)
        robot_frame = Image.fromarray(renderer.render())

        # Load RGB frame
        rgb_frame = Image.open(rgb_files[rgb_idx]).resize((W, H))

        # Composite
        composite = Image.new("RGB", (W * 2, H))
        composite.paste(rgb_frame, (0, 0))
        composite.paste(robot_frame, (W, 0))

        draw = ImageDraw.Draw(composite)
        want_grip = grasping[i % n_traj] > 0
        color = (50, 200, 50) if want_grip else (255, 255, 255)
        phase = "GRASP" if want_grip else "FREE"

        draw.rectangle([(0, 0), (W, 25)], fill=(0, 0, 0))
        draw.text((5, 3), f"iPhone R3D — Frame {rgb_idx}/{n_rgb}", fill=(255, 255, 255))
        draw.rectangle([(W, 0), (W*2, 25)], fill=(0, 0, 0))
        draw.text((W+5, 3), f"H1 Bimanual — {phase} — dex-retargeted", fill=color)
        draw.line([(W, 0), (W, H)], fill=(255, 255, 255), width=2)

        composite.save(fd / f"frame_{i:04d}.png")

        if i % 30 == 0:
            print(f"  F{i:03d}/{n_frames} rgb={rgb_idx}")

    renderer.close()

    ff = shutil.which("ffmpeg") or r"C:\Users\chris\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"
    out = OUT / "stack2_bimanual_sidebyside.mp4"
    subprocess.check_call([ff, "-y", "-framerate", "10", "-i", str(fd/"frame_%04d.png"),
                           "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out)])
    print(f"OK {out}")


if __name__ == "__main__":
    render()
