#!/usr/bin/env python3
"""Side-by-side: R3D capture (left) vs retargeted Shadow Hand (right).

Proves the retargeted data matches the original hand motion.
Left: Your egocentric video frames from the iPhone capture
Right: Shadow Hand with dex-retargeted finger joints, synced frame-by-frame

Output: pipeline/sim_renders/stack2_side_by_side.mp4
"""
import mujoco, numpy as np, json, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import shutil, subprocess

ASSETS = Path(__file__).parent / "humanoidbench_assets"
CALIB = Path(__file__).parent / "wrist_trajectories"
RETARGET = Path(__file__).parent / "retargeted"
RGB_DIR = Path(__file__).parent / "r3d_output" / "stack2_rgb"
OUT = Path(__file__).parent / "sim_renders"
OUT.mkdir(exist_ok=True)

W, H = 640, 480  # per side
FINAL_W = W * 2  # side by side


def render_task(task_name="stack2"):
    # Load retargeted data
    ret = json.loads((RETARGET / f"{task_name}_shadow_hand.json").read_text())
    ret_results = ret["results"]
    joint_names = ret["joint_names"]
    
    # Load wrist trajectory
    calib = json.loads((CALIB / f"{task_name}_calibrated.json").read_text())
    grasping = np.array(calib["grasping"], dtype=float)
    n_traj = len(grasping)
    
    # Get RGB frames
    rgb_files = sorted(RGB_DIR.glob("*.jpg"))
    n_rgb = len(rgb_files)
    print(f"RGB frames: {n_rgb}, Trajectory frames: {n_traj}, Retargeted: {len(ret_results)}")
    
    # Frame mapping: RGB has n_rgb frames, trajectory has n_traj
    # Map trajectory frames to RGB frames (may be different counts)
    # R3D capture at 30fps, trajectory may be subsampled
    
    # Build MuJoCo scene for Shadow Hand ONLY (no humanoid body needed)
    # Use the shadow_hand_menagerie model directly
    # Use H1 model which has Shadow Hand actuators
    sh_path = str(ASSETS / "envs" / "h1hand_pos_cube.xml")
    
    m = mujoco.MjModel.from_xml_path(str(sh_path))
    d = mujoco.MjData(m)
    
    # Check if this has keyframes
    if m.nkey > 0:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    print(f"Shadow Hand model: nq={m.nq} nv={m.nv} nu={m.nu}")
    
    # Build actuator mapping
    act_map = {}
    for ai in range(m.nu):
        an = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, ai) or ""
        # Try matching with and without rh_ prefix
        for prefix in ["", "rh_A_", "A_"]:
            short = an.replace(prefix, "") if prefix else an
            act_map[short] = ai
    print(f"Actuators: {m.nu}")
    
    # If using h1hand, freeze everything except right hand
    if "h1hand" in str(sh_path):
        standing = d.qpos.copy()
        freeze_q, freeze_v = [], []
        for j in range(m.njnt):
            jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or f"_unnamed_{j}"
            if jn.startswith("rh_") or jn.startswith("free_right") or jn.startswith("free_left"):
                continue
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
        use_h1 = True
    else:
        def freeze():
            pass
        use_h1 = False
    
    renderer = mujoco.Renderer(m, H, W)
    fd = OUT / f"_{task_name}_sidebyside_frames"
    if fd.exists():
        shutil.rmtree(fd)
    fd.mkdir()
    
    # Use n_traj as frame count (the trajectory length)
    n_frames = n_traj
    last_joints = None
    
    for i in range(n_frames):
        # Get retargeted frame (may need index mapping)
        # retargeted has 476 frames (every other frame from 30fps R3D)
        # trajectory has 235 frames
        # Map: ret_idx = i * 2 (roughly)
        ret_idx = min(i * 2, len(ret_results) - 1)
        ret_frame = ret_results[ret_idx]
        
        # RGB frame index
        rgb_idx = min(int(i * n_rgb / n_frames), n_rgb - 1)
        
        want_grip = grasping[i] > 0
        
        # Apply retargeted joints to Shadow Hand
        if ret_frame["joints"] is not None:
            joints = ret_frame["joints"]
            for ji, jn in enumerate(joint_names):
                if jn.startswith("dummy_"):
                    continue
                # Try to find matching actuator
                for key in [jn, f"rh_A_{jn}", jn.replace("J", "j")]:
                    if key in act_map:
                        ai = act_map[key]
                        val = np.clip(joints[ji], m.actuator_ctrlrange[ai][0], m.actuator_ctrlrange[ai][1])
                        d.ctrl[ai] = val
                        break
            last_joints = joints
        elif last_joints is not None:
            for ji, jn in enumerate(joint_names):
                if jn.startswith("dummy_"):
                    continue
                for key in [jn, f"rh_A_{jn}"]:
                    if key in act_map:
                        ai = act_map[key]
                        d.ctrl[ai] = np.clip(last_joints[ji], m.actuator_ctrlrange[ai][0], m.actuator_ctrlrange[ai][1])
                        break
        
        # Step physics
        for _ in range(10):
            freeze()
            mujoco.mj_step(m, d)
        
        mujoco.mj_forward(m, d)
        
        # Render Shadow Hand
        if use_h1:
            cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "cam_inhand")
            renderer.update_scene(d, camera=cam_id if cam_id >= 0 else -1)
        else:
            renderer.update_scene(d, camera=-1)
        
        robot_frame = Image.fromarray(renderer.render())
        
        # Load RGB frame
        rgb_path = rgb_files[rgb_idx]
        rgb_frame = Image.open(rgb_path).resize((W, H))
        
        # Composite side by side
        composite = Image.new("RGB", (FINAL_W, H))
        composite.paste(rgb_frame, (0, 0))
        composite.paste(robot_frame, (W, 0))
        
        # Add labels
        draw = ImageDraw.Draw(composite)
        phase = "GRASP" if want_grip else "FREE"
        has_ret = "✓" if ret_frame["joints"] is not None else "○"
        color = (50, 200, 50) if want_grip else (255, 255, 255)
        
        # Left label
        draw.rectangle([(0, 0), (W, 30)], fill=(0, 0, 0, 180))
        draw.text((10, 5), f"iPhone R3D Capture — Frame {rgb_idx}/{n_rgb}", fill=(255, 255, 255))
        
        # Right label
        draw.rectangle([(W, 0), (FINAL_W, 30)], fill=(0, 0, 0, 180))
        draw.text((W + 10, 5), f"Shadow Hand Retargeted {has_ret} — {phase}", fill=color)
        
        # Center divider
        draw.line([(W, 0), (W, H)], fill=(255, 255, 255), width=2)
        
        composite.save(fd / f"frame_{i:04d}.png")
        
        if i % 30 == 0:
            print(f"F{i:03d}/{n_frames} rgb={rgb_idx} ret={ret_idx} grip={want_grip} has_joints={ret_frame['joints'] is not None}")
    
    renderer.close()
    
    # Encode
    ff = shutil.which("ffmpeg") or r"C:\Users\chris\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"
    out = OUT / f"{task_name}_side_by_side.mp4"
    subprocess.check_call([
        ff, "-y", "-framerate", "10",
        "-i", str(fd / "frame_%04d.png"),
        "-vcodec", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "20", "-preset", "fast", str(out)
    ])
    print(f"OK {out}")
    return out


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "stack2"
    render_task(task)
