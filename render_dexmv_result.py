#!/usr/bin/env python3
"""Render DexMV-retargeted joint angles on H1+Shadow Hand in side-by-side view.

DexMV outputs 30-DOF Adroit hand joints. We map the first 20 (finger joints)
to the H1 Shadow Hand actuators (which also has 20 finger actuators per hand).
"""
import mujoco, numpy as np, json
from pathlib import Path
from PIL import Image, ImageDraw
import shutil, subprocess

ASSETS = Path(__file__).parent / "humanoidbench_assets"
RETARGET = Path(__file__).parent / "retargeted"
RGB_DIR = Path(__file__).parent / "r3d_output" / "stack2_rgb"
OUT = Path(__file__).parent / "sim_renders"
OUT.mkdir(exist_ok=True)

W, H = 480, 360


def render():
    # Load DexMV results
    ret_r = json.loads((RETARGET / "stack2_right_dexmv.json").read_text())
    ret_l = json.loads((RETARGET / "stack2_left_dexmv.json").read_text())
    arr_r = np.array(ret_r["robot_joints"])  # (476, 30)
    arr_l = np.array(ret_l["robot_joints"])
    n_frames = arr_r.shape[0]
    print(f"DexMV: {n_frames} frames, right={arr_r.shape[1]}dof, left={arr_l.shape[1]}dof")

    # RGB frames
    rgb_files = sorted(RGB_DIR.glob("*.jpg"))
    n_rgb = len(rgb_files)

    # Build MuJoCo scene
    spec = mujoco.MjSpec.from_file(str(ASSETS / "envs" / "h1hand_pos_cube.xml"))
    table = spec.worldbody.add_body()
    table.name = "table"
    table.pos = np.array([0.45, 0.0, 0.475])
    tg = table.add_geom()
    tg.type = mujoco.mjtGeom.mjGEOM_BOX
    tg.size = np.array([0.25, 0.35, 0.475])
    tg.rgba = np.array([0.55, 0.35, 0.18, 1.0])
    tg.mass = 100.0; tg.contype = 1; tg.conaffinity = 1

    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)

    # Fix wrist yaw = 0 (no twist)
    for jn in ["left_wrist_yaw", "right_wrist_yaw"]:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid >= 0:
            d.qpos[m.jnt_qposadr[jid]] = 0.0

    mujoco.mj_forward(m, d)
    standing = d.qpos.copy()

    # Get actuator maps
    rh_acts_ordered = []  # in order of Shadow Hand actuator index
    lh_acts_ordered = []
    for ai in range(m.nu):
        an = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, ai) or ""
        if an.startswith("rh_A_"):
            rh_acts_ordered.append(ai)
        elif an.startswith("lh_A_"):
            lh_acts_ordered.append(ai)
    print(f"Shadow Hand actuators: rh={len(rh_acts_ordered)}, lh={len(lh_acts_ordered)}")

    # Freeze body
    free_set = set()
    for j in range(m.njnt):
        jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if jn.startswith("rh_") or jn.startswith("lh_") or "cube" in jn or \
           jn.startswith("right_") or jn.startswith("left_"):
            free_set.add(jn)

    freeze_q, freeze_v = [], []
    for j in range(m.njnt):
        jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or f"_u{j}"
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
    for _ in range(100):
        freeze()
        mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)

    # Place books
    pick_jnt = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "free_right_cube_to_rotate")
    sup_jnt = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "free_left_cube_to_rotate")
    pick_qa = m.jnt_qposadr[pick_jnt]
    sup_qa = m.jnt_qposadr[sup_jnt]
    book_z = 0.95 + 0.012 + 0.001
    for gi in range(m.ngeom):
        if m.geom_bodyid[gi] in [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in ["left_cube_to_rotate","right_cube_to_rotate"]]:
            m.geom_size[gi] = [0.09, 0.06, 0.012]

    d.qpos[pick_qa:pick_qa+3] = [0.40, -0.08, book_z]
    d.qpos[pick_qa+3:pick_qa+7] = [1, 0, 0, 0]
    d.qpos[sup_qa:sup_qa+3] = [0.40, 0.08, book_z]
    d.qpos[sup_qa+3:sup_qa+7] = [1, 0, 0, 0]
    for jnt in [pick_jnt, sup_jnt]:
        d.qvel[m.jnt_dofadr[jnt]:m.jnt_dofadr[jnt]+6] = 0
    mujoco.mj_forward(m, d)

    # Arm IK from pixel wrist tracking
    gpu_hands = json.loads((Path(__file__).parent / "modal_results" / "stack2_gpu_hands.json").read_text())
    left_wp = [None] * len(gpu_hands["results"])
    right_wp = [None] * len(gpu_hands["results"])
    for ri, r in enumerate(gpu_hands["results"]):
        for h in r.get("hands", []):
            wp = h.get("wrist_pixel")
            if not wp: continue
            side = h.get("hand", "unknown")
            if side == "left": left_wp[ri] = wp
            elif side == "right": right_wp[ri] = wp
            elif wp[0] > 480: left_wp[ri] = wp
            else: right_wp[ri] = wp

    # Smooth pixel trajectories
    from scipy.signal import savgol_filter
    img_w, img_h = 960, 720
    def smooth_pix(wp_list, window=15):
        arr = np.array([wp if wp is not None else [img_w/2, img_h/2] for wp in wp_list])
        w = min(window, len(arr)//2*2-1)
        if w < 5: w = 5
        for ax in range(2):
            arr[:, ax] = savgol_filter(arr[:, ax], w, 3)
        return arr
    
    l_pix = smooth_pix(left_wp)
    r_pix = smooth_pix(right_wp)
    l_pix_center = l_pix.mean(axis=0)
    r_pix_center = r_pix.mean(axis=0)
    pix_scale = 0.15 / 350.0

    # Arm joints + IK
    arm_names_r = ["right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow"]
    arm_names_l = ["left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow"]
    arm_qa_r = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)] for jn in arm_names_r]
    arm_qa_l = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)] for jn in arm_names_l]
    arm_da_r = [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)] for jn in arm_names_r]
    arm_da_l = [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)] for jn in arm_names_l]
    arm_jids_r = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn) for jn in arm_names_r]
    arm_jids_l = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn) for jn in arm_names_l]
    arm_lo_r = np.array([m.jnt_range[j][0] for j in arm_jids_r])
    arm_hi_r = np.array([m.jnt_range[j][1] for j in arm_jids_r])
    arm_lo_l = np.array([m.jnt_range[j][0] for j in arm_jids_l])
    arm_hi_l = np.array([m.jnt_range[j][1] for j in arm_jids_l])
    rh_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_hand")
    lh_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "left_hand")
    rh_default = d.site_xpos[rh_sid].copy()
    lh_default = d.site_xpos[lh_sid].copy()

    arm_acts_r = {an: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, an) for an in arm_names_r if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, an) >= 0}
    arm_acts_l = {an: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, an) for an in arm_names_l if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, an) >= 0}

    jac = np.zeros((3, m.nv))
    jac_r = np.zeros((3, m.nv))
    def ik_arm(target, site_id, arm_qa, arm_da, arm_lo, arm_hi, max_iter=40):
        for _ in range(max_iter):
            mujoco.mj_forward(m, d)
            pos = d.site_xpos[site_id].copy()
            err = target - pos
            if np.linalg.norm(err) < 0.005: break
            mujoco.mj_jacSite(m, d, jac, jac_r, site_id)
            J = jac[:, arm_da]
            JJT = J @ J.T + 0.005 * np.eye(3)
            dq = J.T @ np.linalg.solve(JJT, err)
            for ii, qa in enumerate(arm_qa):
                d.qpos[qa] = np.clip(d.qpos[qa] + dq[ii] * 0.3, arm_lo[ii], arm_hi[ii])

    print(f"Arm IK ready: rh={len(arm_acts_r)} lh={len(arm_acts_l)} actuators")

    # Renderer
    renderer = mujoco.Renderer(m, H, W)
    fd = OUT / "_dexmv_sidebyside_frames"
    if fd.exists(): shutil.rmtree(fd)
    fd.mkdir()

    # DexMV Adroit→H1 Shadow Hand joint name mapping
    # Adroit 30 joints: [ARTx,y,z(0-2), ARRx,y,z(3-5), WRJ1(6), WRJ0(7),
    #   FFJ3(8),FFJ2(9),FFJ1(10),FFJ0(11), MFJ3(12)..MFJ0(15), RFJ3(16)..RFJ0(19),
    #   LFJ4(20),LFJ3(21)..LFJ0(24), THJ4(25),THJ3(26),THJ2(27),THJ1(28),THJ0(29)]
    # Shadow Hand actuators: rh_A_WRJ2, rh_A_WRJ1, rh_A_THJ5..1, rh_A_FFJ4,FFJ3,FFJ0, etc.
    
    # Build name-based mapping
    adroit_joint_names = [
        "ARTx","ARTy","ARTz","ARRx","ARRy","ARRz",
        "WRJ1","WRJ0",
        "FFJ3","FFJ2","FFJ1","FFJ0",
        "MFJ3","MFJ2","MFJ1","MFJ0",
        "RFJ3","RFJ2","RFJ1","RFJ0",
        "LFJ4","LFJ3","LFJ2","LFJ1","LFJ0",
        "THJ4","THJ3","THJ2","THJ1","THJ0",
    ]
    
    # Map Adroit joint names → Shadow Hand actuator indices
    rh_name_map = {}  # adroit_idx → shadow_actuator_idx
    lh_name_map = {}
    for ai in range(m.nu):
        an = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, ai) or ""
        if an.startswith("rh_A_"):
            sh_name = an[5:]  # strip "rh_A_"
            for adroit_idx, adroit_name in enumerate(adroit_joint_names):
                if sh_name == adroit_name:
                    rh_name_map[adroit_idx] = ai
                    break
            # Also match WRJ2→WRJ1, WRJ1→WRJ0 (different naming)
            if sh_name == "WRJ2":
                rh_name_map[6] = ai  # Adroit WRJ1 → Shadow WRJ2
            elif sh_name == "WRJ1":
                rh_name_map[7] = ai  # Adroit WRJ0 → Shadow WRJ1
            # Shadow uses THJ5 instead of THJ4
            elif sh_name == "THJ5":
                rh_name_map[25] = ai  # Adroit THJ4 → Shadow THJ5
            # Shadow uses FFJ0 (coupled) instead of FFJ0+FFJ1
            elif sh_name == "FFJ0":
                rh_name_map[11] = ai  # Use FFJ0
            elif sh_name == "MFJ0":
                rh_name_map[15] = ai
            elif sh_name == "RFJ0":
                rh_name_map[19] = ai
            elif sh_name == "LFJ0":
                rh_name_map[24] = ai
            elif sh_name == "LFJ5":
                rh_name_map[20] = ai  # Adroit LFJ4 → Shadow LFJ5
        elif an.startswith("lh_A_"):
            sh_name = an[5:]
            for adroit_idx, adroit_name in enumerate(adroit_joint_names):
                if sh_name == adroit_name:
                    lh_name_map[adroit_idx] = ai
                    break
            if sh_name == "WRJ2": lh_name_map[6] = ai
            elif sh_name == "WRJ1": lh_name_map[7] = ai
            elif sh_name == "THJ5": lh_name_map[25] = ai
            elif sh_name == "FFJ0": lh_name_map[11] = ai
            elif sh_name == "MFJ0": lh_name_map[15] = ai
            elif sh_name == "RFJ0": lh_name_map[19] = ai
            elif sh_name == "LFJ0": lh_name_map[24] = ai
            elif sh_name == "LFJ5": lh_name_map[20] = ai
    
    print(f"Joint mapping: rh={len(rh_name_map)} adroit→shadow, lh={len(lh_name_map)}")

    n_traj = min(n_frames, n_rgb)
    for i in range(n_traj):
        ret_idx = min(i * 2, n_frames - 1)
        rgb_idx = min(i, n_rgb - 1)

        # Arm IK from pixel tracking
        wp_idx = min(i * 2, len(r_pix) - 1)
        # Right arm target
        rpx, rpy = r_pix[wp_idx]
        target_r = rh_default.copy()
        target_r[1] += (rpx - r_pix_center[0]) * pix_scale
        target_r[2] += -(rpy - r_pix_center[1]) * pix_scale
        ik_arm(target_r, rh_sid, arm_qa_r, arm_da_r, arm_lo_r, arm_hi_r)
        for an, ai in arm_acts_r.items():
            idx = arm_names_r.index(an)
            d.ctrl[ai] = d.qpos[arm_qa_r[idx]]
        
        # Left arm target
        lpx, lpy = l_pix[min(wp_idx, len(l_pix)-1)]
        target_l = lh_default.copy()
        target_l[1] += -(lpx - l_pix_center[0]) * pix_scale
        target_l[2] += -(lpy - l_pix_center[1]) * pix_scale
        ik_arm(target_l, lh_sid, arm_qa_l, arm_da_l, arm_lo_l, arm_hi_l)
        for an, ai in arm_acts_l.items():
            idx = arm_names_l.index(an)
            d.ctrl[ai] = d.qpos[arm_qa_l[idx]]

        # Apply right hand DexMV joints via name-based mapping
        for adroit_idx, shadow_ai in rh_name_map.items():
            val = arr_r[ret_idx, adroit_idx]
            d.ctrl[shadow_ai] = np.clip(val, m.actuator_ctrlrange[shadow_ai][0], m.actuator_ctrlrange[shadow_ai][1])

        # Apply left hand DexMV joints
        for adroit_idx, shadow_ai in lh_name_map.items():
            val = arr_l[ret_idx, adroit_idx]
            d.ctrl[shadow_ai] = np.clip(val, m.actuator_ctrlrange[shadow_ai][0], m.actuator_ctrlrange[shadow_ai][1])

        # Step physics
        for _ in range(15):
            freeze()
            mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)

        # Render
        cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "cam_inhand")
        renderer.update_scene(d, camera=cam_id if cam_id >= 0 else -1)
        robot_frame = Image.fromarray(renderer.render())

        # Load RGB
        rgb_frame = Image.open(rgb_files[rgb_idx]).resize((W, H))

        # Composite
        composite = Image.new("RGB", (W * 2, H))
        composite.paste(rgb_frame, (0, 0))
        composite.paste(robot_frame, (W, 0))

        draw = ImageDraw.Draw(composite)
        draw.rectangle([(0, 0), (W, 22)], fill=(0, 0, 0))
        draw.text((5, 3), f"iPhone R3D — F{rgb_idx}/{n_rgb}", fill=(255, 255, 255))
        draw.rectangle([(W, 0), (W*2, 22)], fill=(0, 0, 0))
        draw.text((W+5, 3), f"DexMV Adroit→H1 Shadow — F{i}", fill=(100, 255, 100))
        draw.line([(W, 0), (W, H)], fill=(255, 255, 255), width=2)

        composite.save(fd / f"frame_{i:04d}.png")

        if i % 30 == 0:
            print(f"  F{i:03d}/{n_traj}")

    renderer.close()

    ff = shutil.which("ffmpeg") or r"C:\Users\chris\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"
    out = OUT / "stack2_dexmv_sidebyside.mp4"
    subprocess.check_call([ff, "-y", "-framerate", "10", "-i", str(fd/"frame_%04d.png"),
                           "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out)])
    print(f"OK {out}")


if __name__ == "__main__":
    render()
