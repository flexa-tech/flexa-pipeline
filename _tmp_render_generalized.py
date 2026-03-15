"""
Generalized GT vs HaMeR replay render for any EgoDex episode.
Uses pick-and-place calibration (affine transform) on new episodes.
"""
import json, h5py, numpy as np, mujoco, cv2, os, sys
import tempfile, shutil, subprocess
from pathlib import Path
from scipy.ndimage import uniform_filter1d
from replay_egodex_on_h1 import (
    load_egodex_episode, egodex_to_robot_frame, MuJoCoArmIK,
    ARM_JOINTS, FINGER_CURL_JOINTS, fingertip_distance_to_curl
)
from numpy.linalg import lstsq

ASSETS_DIR = Path(__file__).parent / "humanoidbench_assets"
WIDTH, HEIGHT = 480, 360

# ===== CALIBRATION: derived from pick-and-place =====
# Load pick-and-place calibration data to get the affine transforms
print("Computing calibration from pick-and-place...")
with open('egodex_sample/hamer_pick_place.json') as f:
    cal_hamer = json.load(f)
cal_gt = h5py.File('egodex_sample/test/basic_pick_place/0.hdf5', 'r')
cal_cam = cal_gt['transforms']['camera'][()]
cal_n = cal_cam.shape[0]
focal = 736.6339; img_h = 1080

A_b_cal = {}
for side in ['right', 'left']:
    h_cam, g_cam = [], []
    for fi in range(cal_n):
        hands = cal_hamer[str(fi)].get('hands', [])
        sh = [h for h in hands if h['side'] == side]
        if sh:
            s, tx, ty = sh[0]['cam_t']
            kps = np.array(sh[0]['keypoints_3d'])
            z = 2*focal/(s*img_h)
            h_cam.append([tx/s + kps[0,0]*z, ty/s + kps[0,1]*z, z + kps[0,2]*z])
            cam_inv = np.linalg.inv(cal_cam[fi])
            g_cam.append((cam_inv @ cal_gt['transforms'][f'{side}Hand'][fi])[:3, 3])
    h_cam = np.array(h_cam); g_cam = np.array(g_cam)
    H_aug = np.column_stack([h_cam, np.ones(len(h_cam))])
    A_b_cal[side], _, _, _ = lstsq(H_aug, g_cam, rcond=None)
    pred = H_aug @ A_b_cal[side]
    for ax, nm in enumerate(['X','Y','Z']):
        corr = np.corrcoef(pred[:,ax], g_cam[:,ax])[0,1]
        print(f"  {side} {nm}: cal corr={corr:.3f}")
cal_gt.close()


def process_episode(episode_type, episode_idx=0):
    """Full pipeline: load GT + HaMeR, apply calibration, render comparison."""
    print(f"\n===== Processing {episode_type}/{episode_idx} =====")
    
    # Load GT
    gt_data = load_egodex_episode(episode_type, episode_idx)
    gt_converted = egodex_to_robot_frame(gt_data)
    n_frames = gt_data['n_frames']
    
    # Load HaMeR
    hamer_path = f'egodex_sample/hamer_{episode_type}.json'
    with open(hamer_path) as f:
        hamer = json.load(f)
    
    # Load episode GT for camera transforms
    task_map = {'pick_place': 'basic_pick_place', 'fold': 'basic_fold', 'stack': 'stack'}
    gt_file = h5py.File(f'egodex_sample/test/{task_map[episode_type]}/{episode_idx}.hdf5', 'r')
    cam_transforms = gt_file['transforms']['camera'][()]
    
    # Reconstruct HaMeR 3D + apply CALIBRATION transforms
    hamer_data = {
        'n_frames': n_frames,
        'description': gt_data['description'],
        'camera': gt_data['camera'],
    }
    
    for side in ['left', 'right']:
        hamer_cam = []
        for fi in range(n_frames):
            hands = hamer.get(str(fi), {}).get('hands', [])
            sh = [h for h in hands if h['side'] == side]
            if sh:
                s, tx, ty = sh[0]['cam_t']
                kps = np.array(sh[0]['keypoints_3d'])
                z = 2*focal/(s*img_h)
                hamer_cam.append([tx/s + kps[0,0]*z, ty/s + kps[0,1]*z, z + kps[0,2]*z])
            else:
                hamer_cam.append([np.nan]*3)
        hamer_cam = np.array(hamer_cam)
        
        # Apply calibration affine transform
        valid = ~np.isnan(hamer_cam[:,0])
        hamer_gt_cam = np.full_like(hamer_cam, np.nan)
        if valid.sum() > 0:
            H_aug = np.column_stack([hamer_cam[valid], np.ones(valid.sum())])
            hamer_gt_cam[valid] = H_aug @ A_b_cal[side]
        
        # Interpolate NaN
        for ax in range(3):
            vals = hamer_gt_cam[:, ax]
            nans = np.isnan(vals)
            if nans.any() and (~nans).sum() > 2:
                vals[nans] = np.interp(np.where(nans)[0], np.where(~nans)[0], vals[~nans])
        
        # Adaptive smoothing
        gt_cam_side = []
        for fi in range(n_frames):
            cam_inv = np.linalg.inv(cam_transforms[fi])
            gt_cam_side.append((cam_inv @ gt_file['transforms'][f'{side}Hand'][fi])[:3, 3])
        gt_cam_side = np.array(gt_cam_side)
        
        travel = np.sum(np.linalg.norm(np.diff(hamer_gt_cam[valid], axis=0), axis=1)) if valid.sum() > 1 else 0
        gt_travel = np.sum(np.linalg.norm(np.diff(gt_cam_side, axis=0), axis=1))
        ratio = travel / (gt_travel + 1e-6)
        
        if ratio > 5.0:
            median_pos = np.median(hamer_gt_cam[valid], axis=0) if valid.sum() > 0 else np.zeros(3)
            for fi in range(n_frames): hamer_gt_cam[fi] = median_pos
            print(f"  {side}: ratio={ratio:.1f}x → FREEZE")
        else:
            sw = 9
            for ax in range(3):
                hamer_gt_cam[:, ax] = uniform_filter1d(hamer_gt_cam[:, ax], size=sw)
            print(f"  {side}: ratio={ratio:.1f}x → smooth(w={sw})")
        
        # Camera → world
        wrist_world = np.zeros((n_frames, 3))
        for fi in range(n_frames):
            cam_tf = cam_transforms[fi]
            wrist_world[fi] = cam_tf[:3,:3] @ hamer_gt_cam[fi] + cam_tf[:3,3]
        
        hamer_data[f'{side}_wrist_pos'] = wrist_world
        hamer_data[f'{side}_wrist_rot'] = gt_data[f'{side}_wrist_rot']
        hamer_data[f'{side}_fingertips'] = gt_data[f'{side}_fingertips']  # Use GT fingertips for position
    
    gt_file.close()
    
    if 'mp4_path' in gt_data:
        hamer_data['mp4_path'] = gt_data['mp4_path']
    
    hamer_converted = egodex_to_robot_frame(hamer_data)
    
    # Load DexMV
    dexmv_data = None
    for dexmv_prefix in [f'egodex_hamer_{episode_type}', f'egodex_hamer']:
        r_path = f'retargeted/{dexmv_prefix}_right_dexmv.json'
        l_path = f'retargeted/{dexmv_prefix}_left_dexmv.json'
        if os.path.exists(r_path) and os.path.exists(l_path):
            with open(r_path) as f: r = json.load(f)
            with open(l_path) as f: l = json.load(f)
            dexmv_data = {'right': np.array(r['robot_joints']), 'left': np.array(l['robot_joints'])}
            # Smooth DexMV fingers
            for s in ['right', 'left']:
                for dof in range(6, 30):
                    dexmv_data[s][:, dof] = uniform_filter1d(dexmv_data[s][:, dof], size=11)
            print(f"  DexMV loaded from {dexmv_prefix}")
            break
    
    # Print correlation
    print(f"  MuJoCo ranges:")
    for src, data in [("GT", gt_converted), ("HaMeR", hamer_converted)]:
        rp = data['right_wrist_pos']
        print(f"    {src} right: X=[{rp[:,0].min():.3f},{rp[:,0].max():.3f}] Y=[{rp[:,1].min():.3f},{rp[:,1].max():.3f}]")
    
    # ===== RENDER =====
    scene_xml = f"""
    <mujoco>
      <include file="{str(ASSETS_DIR / 'robots' / 'h1hand_pos.xml')}"/>
      <visual><global offwidth="{WIDTH}" offheight="{HEIGHT}"/></visual>
      <asset>
        <texture type="2d" name="grid" builtin="checker" rgb1="0.9 0.9 0.9" rgb2="0.7 0.7 0.7" width="512" height="512"/>
        <material name="floor_mat" texture="grid" texrepeat="8 8"/>
      </asset>
      <worldbody>
        <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
        <light pos="1 1 2" dir="-0.5 -0.5 -1" diffuse="0.4 0.4 0.4"/>
        <geom name="floor" type="plane" size="3 3 0.1" material="floor_mat"/>
      </worldbody>
    </mujoco>
    """
    m_gt = mujoco.MjModel.from_xml_string(scene_xml)
    d_gt = mujoco.MjData(m_gt)
    m_hm = mujoco.MjModel.from_xml_string(scene_xml)
    d_hm = mujoco.MjData(m_hm)
    renderer = mujoco.Renderer(m_gt, width=WIDTH, height=HEIGHT)
    
    joint_map = {}
    for i in range(m_gt.njnt):
        joint_map[m_gt.joint(i).name] = m_gt.jnt_qposadr[i]
    
    ik_gt = MuJoCoArmIK(m_gt, d_gt, joint_map)
    ik_hm = MuJoCoArmIK(m_hm, d_hm, joint_map)
    
    orig_frames = []
    if 'mp4_path' in gt_data:
        cap = cv2.VideoCapture(gt_data['mp4_path'])
        while True:
            ret, frame = cap.read()
            if not ret: break
            orig_frames.append(frame)
        cap.release()
    
    adroit_to_shadow = {
        6: 'WRJ1', 7: 'WRJ2',
        8: 'FFJ3', 9: 'FFJ2', 10: 'FFJ1',
        12: 'MFJ3', 13: 'MFJ2', 14: 'MFJ1',
        16: 'RFJ3', 17: 'RFJ2', 18: 'RFJ1',
        21: 'LFJ3', 22: 'LFJ2', 23: 'LFJ1',
        25: 'THJ5', 26: 'THJ4', 27: 'THJ3', 28: 'THJ2', 29: 'THJ1',
    }
    
    def render_robot(model, data, ik_solver, converted, fi, use_dexmv=False):
        data.qpos[0:3] = [0, 0, 0.98]
        data.qpos[3:7] = [1, 0, 0, 0]
        for lj in ['left_hip_pitch', 'right_hip_pitch']:
            if lj in joint_map: data.qpos[joint_map[lj]] = -0.1
        for kj in ['left_knee', 'right_knee']:
            if kj in joint_map: data.qpos[joint_map[kj]] = 0.2
        
        for side in ['left', 'right']:
            target = converted[f'{side}_wrist_pos'][fi]
            ik_solver.solve(target, side, n_iter=200, step_size=0.5)
            for ji, jname in enumerate(ARM_JOINTS[side]):
                if jname in joint_map:
                    data.qpos[joint_map[jname]] = data.qpos[ik_solver.arm_qpos_idx[side][ji]]
        
        if use_dexmv and dexmv_data is not None and fi < dexmv_data['right'].shape[0]:
            for side in ['left', 'right']:
                prefix = 'lh' if side == 'left' else 'rh'
                dj = dexmv_data[side][fi]
                for aidx, ssuffix in adroit_to_shadow.items():
                    jname = f'{prefix}_{ssuffix}'
                    if jname in joint_map:
                        jid = model.joint(jname).id
                        lo, hi = model.jnt_range[jid]
                        data.qpos[joint_map[jname]] = np.clip(dj[aidx], lo, hi)
        else:
            for side in ['left', 'right']:
                wp = converted[f'{side}_wrist_pos'][fi]
                ft = converted[f'{side}_fingertips'][fi]
                for fi2, fn in enumerate(['index','middle','ring','little','thumb']):
                    curl = fingertip_distance_to_curl(wp, ft[fi2])
                    for cj in FINGER_CURL_JOINTS[side][fn]:
                        if cj in joint_map:
                            jid = model.joint(cj).id
                            lo, hi = model.jnt_range[jid]
                            data.qpos[joint_map[cj]] = lo + curl*(hi-lo)*0.9
        
        mujoco.mj_forward(model, data)
        
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [0.3, 0, 1.05]
        cam.distance = 1.6; cam.azimuth = 180; cam.elevation = -8
        opt = mujoco.MjvOption()
        renderer.update_scene(data, camera=cam, scene_option=opt)
        renderer._scene.lights[0].pos[:] = [0,0,4]
        renderer._scene.lights[0].dir[:] = [0,0,-1]
        renderer._scene.lights[0].diffuse[:] = [0.9,0.9,0.9]
        renderer._scene.lights[0].ambient[:] = [0.6,0.6,0.6]
        if renderer._scene.nlight > 1:
            renderer._scene.lights[1].diffuse[:] = [0.7,0.7,0.7]
            renderer._scene.lights[1].ambient[:] = [0.4,0.4,0.4]
        return cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
    
    # Init
    for dd in [d_gt, d_hm]:
        mujoco.mj_resetData(m_gt, dd)
        dd.qpos[0:3] = [0,0,0.98]; dd.qpos[3:7] = [1,0,0,0]
        mujoco.mj_forward(m_gt, dd)
    
    tmpdir = tempfile.mkdtemp(prefix=f"render_{episode_type}_")
    print(f"  Rendering {n_frames} frames...")
    
    for fi in range(n_frames):
        gt_frame = render_robot(m_gt, d_gt, ik_gt, gt_converted, fi, use_dexmv=False)
        hamer_frame = render_robot(m_hm, d_hm, ik_hm, hamer_converted, fi, use_dexmv=True)
        
        orig = cv2.resize(orig_frames[min(fi, len(orig_frames)-1)], (WIDTH, HEIGHT)) if orig_frames else np.zeros((HEIGHT,WIDTH,3), dtype=np.uint8)
        
        for img, label in [(orig, "Original"), (gt_frame, "GT Replay"), (hamer_frame, "Pipeline Replay")]:
            cv2.rectangle(img, (5,5), (230,35), (0,0,0), -1)
            cv2.putText(img, label, (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        combined = np.hstack([orig, gt_frame, hamer_frame])
        cv2.imwrite(os.path.join(tmpdir, f"frame_{fi:05d}.png"), combined)
        if fi % 60 == 0: print(f"    Frame {fi}/{n_frames}")
    
    renderer.close()
    
    output = f"sim_renders/egodex_{episode_type}_pipeline_test.mp4"
    subprocess.run(["ffmpeg","-y","-framerate","30","-i",os.path.join(tmpdir,"frame_%05d.png"),
                    "-c:v","libx264","-pix_fmt","yuv420p","-crf","23",output], capture_output=True)
    shutil.rmtree(tmpdir)
    print(f"  Saved: {output}")


# Run on fold and stack
for ep in ['fold', 'stack']:
    process_episode(ep)
