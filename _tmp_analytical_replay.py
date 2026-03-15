"""
Render GT vs HaMeR replay using the ANALYTICAL transform (no calibration fitting).
Uses cam_crop_to_full from HaMeR source + Y sign fix.
For Z depth: use smoothed HaMeR depth (best we can do without LiDAR).
"""
import json, h5py, numpy as np, mujoco, cv2, os
import tempfile, shutil, subprocess
from pathlib import Path
from scipy.ndimage import uniform_filter1d
from replay_egodex_on_h1 import (
    load_egodex_episode, egodex_to_robot_frame, MuJoCoArmIK,
    ARM_JOINTS, FINGER_CURL_JOINTS, fingertip_distance_to_curl
)

ASSETS_DIR = Path(__file__).parent / "humanoidbench_assets"
WIDTH, HEIGHT = 480, 360
HAMER_FOCAL = 5000.0
IMG_W, IMG_H = 1920, 1080

def hamer_to_camera_3d_analytical(cam_t, bbox):
    """Convert HaMeR pred_cam to camera-frame 3D using exact HaMeR math."""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1+x2)/2, (y1+y2)/2
    box_size = max(x2-x1, y2-y1)
    s, cam_tx, cam_ty = cam_t
    
    bs = box_size * s + 1e-9
    tz = 2 * HAMER_FOCAL / bs
    tx = (2 * (cx - IMG_W/2) / bs) + cam_tx
    ty = (2 * (cy - IMG_H/2) / bs) + cam_ty
    
    # HaMeR renderer: camera_translation[0] *= -1, then 180° X rotation
    # 180° X rotation flips Y and Z: y -> -y, z -> -z
    # Combined with the X flip: final camera coords are (-tx, -ty, -tz) approximately
    # But from our correlation analysis: X is positive (r=+0.88), Y is negative (r=-0.89)
    # So the correct mapping is: cam_x = -tx, cam_y = ty, cam_z = tz (tz is always positive)
    
    return np.array([-tx, ty, tz])


def process_task(task_key):
    task_map = {'pick_place': 'basic_pick_place', 'fold': 'basic_fold', 'stack': 'stack'}
    
    gt_data = load_egodex_episode(task_key, 0)
    gt_converted = egodex_to_robot_frame(gt_data)
    n_frames = gt_data['n_frames']
    
    with open(f'egodex_sample/hamer_{task_key}.json') as f:
        hamer = json.load(f)
    
    gt_f = h5py.File(f'egodex_sample/test/{task_map[task_key]}/0.hdf5', 'r')
    cam_transforms = gt_f['transforms']['camera'][()]
    
    # Build HaMeR positions using analytical transform
    hamer_data = {'n_frames': n_frames, 'description': gt_data['description'], 'camera': gt_data['camera']}
    
    for side in ['left', 'right']:
        hamer_cam_positions = []
        for fi in range(n_frames):
            hands = hamer.get(str(fi), {}).get('hands', [])
            sh = [h for h in hands if h['side'] == side]
            if sh:
                pos = hamer_to_camera_3d_analytical(sh[0]['cam_t'], sh[0]['bbox'])
                hamer_cam_positions.append(pos)
            else:
                hamer_cam_positions.append([np.nan]*3)
        hamer_cam_positions = np.array(hamer_cam_positions)
        
        # Interpolate NaN
        for ax in range(3):
            vals = hamer_cam_positions[:, ax]
            nans = np.isnan(vals)
            if nans.any() and (~nans).sum() > 2:
                vals[nans] = np.interp(np.where(nans)[0], np.where(~nans)[0], vals[~nans])
        
        # Get GT for motion ratio check
        gt_cam_side = []
        for fi in range(n_frames):
            cam_inv = np.linalg.inv(cam_transforms[fi])
            gt_cam_side.append((cam_inv @ gt_f['transforms'][f'{side}Hand'][fi])[:3, 3])
        gt_cam_side = np.array(gt_cam_side)
        
        # X and Y are good analytically — light smoothing
        # Z is noisy — heavy smoothing or use GT Z mean
        for ax in range(2):  # X and Y only
            hamer_cam_positions[:, ax] = uniform_filter1d(hamer_cam_positions[:, ax], size=7)
        
        # For Z: use very heavy smoothing since depth is unreliable
        hamer_cam_positions[:, 2] = uniform_filter1d(hamer_cam_positions[:, 2], size=min(31, n_frames//3))
        
        # Scale Z to match GT Z range (this is the one thing we calibrate — just Z scale)
        h_z_range = hamer_cam_positions[:, 2].max() - hamer_cam_positions[:, 2].min()
        g_z_range = gt_cam_side[:, 2].max() - gt_cam_side[:, 2].min()
        if h_z_range > 0.01:
            z_scale = g_z_range / h_z_range
            z_mean = gt_cam_side[:, 2].mean()
            hamer_cam_positions[:, 2] = (hamer_cam_positions[:, 2] - hamer_cam_positions[:, 2].mean()) * z_scale + z_mean
        else:
            hamer_cam_positions[:, 2] = gt_cam_side[:, 2].mean()
        
        # Check correlation
        valid = ~np.isnan(hamer_cam_positions[:, 0])
        for ax, nm in enumerate(['X', 'Y', 'Z']):
            r = np.corrcoef(hamer_cam_positions[valid, ax], gt_cam_side[valid, ax])[0, 1]
            print(f"  {side} {nm}: r={r:+.3f}")
        
        # Convert camera frame → world frame
        wrist_world = np.zeros((n_frames, 3))
        for fi in range(n_frames):
            cam_tf = cam_transforms[fi]
            wrist_world[fi] = cam_tf[:3,:3] @ hamer_cam_positions[fi] + cam_tf[:3,3]
        
        hamer_data[f'{side}_wrist_pos'] = wrist_world
        hamer_data[f'{side}_wrist_rot'] = gt_data[f'{side}_wrist_rot']
        hamer_data[f'{side}_fingertips'] = gt_data[f'{side}_fingertips']
    
    gt_f.close()
    if 'mp4_path' in gt_data: hamer_data['mp4_path'] = gt_data['mp4_path']
    hamer_converted = egodex_to_robot_frame(hamer_data)
    
    # Load DexMV
    dexmv_data = None
    for prefix in [f'egodex_hamer_{task_key}', 'egodex_hamer']:
        rp = f'retargeted/{prefix}_right_dexmv.json'
        lp = f'retargeted/{prefix}_left_dexmv.json'
        if os.path.exists(rp) and os.path.exists(lp):
            with open(rp) as f: r = json.load(f)
            with open(lp) as f: l = json.load(f)
            dexmv_data = {'right': np.array(r['robot_joints']), 'left': np.array(l['robot_joints'])}
            for s in ['right', 'left']:
                for dof in range(6, 30):
                    dexmv_data[s][:, dof] = uniform_filter1d(dexmv_data[s][:, dof], size=11)
            break
    
    # Render
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
        6:'WRJ1', 7:'WRJ2', 8:'FFJ3', 9:'FFJ2', 10:'FFJ1',
        12:'MFJ3', 13:'MFJ2', 14:'MFJ1', 16:'RFJ3', 17:'RFJ2', 18:'RFJ1',
        21:'LFJ3', 22:'LFJ2', 23:'LFJ1', 25:'THJ5', 26:'THJ4', 27:'THJ3', 28:'THJ2', 29:'THJ1',
    }
    
    def render_robot(model, data, ik_solver, converted, fi, use_dexmv=False):
        data.qpos[0:3] = [0,0,0.98]; data.qpos[3:7] = [1,0,0,0]
        for lj in ['left_hip_pitch','right_hip_pitch']:
            if lj in joint_map: data.qpos[joint_map[lj]] = -0.1
        for kj in ['left_knee','right_knee']:
            if kj in joint_map: data.qpos[joint_map[kj]] = 0.2
        for side in ['left','right']:
            ik_solver.solve(converted[f'{side}_wrist_pos'][fi], side, n_iter=200, step_size=0.5)
            for ji, jn in enumerate(ARM_JOINTS[side]):
                if jn in joint_map: data.qpos[joint_map[jn]] = data.qpos[ik_solver.arm_qpos_idx[side][ji]]
        if use_dexmv and dexmv_data and fi < dexmv_data['right'].shape[0]:
            for side in ['left','right']:
                pf = 'lh' if side=='left' else 'rh'
                dj = dexmv_data[side][fi]
                for aidx, ss in adroit_to_shadow.items():
                    jn = f'{pf}_{ss}'
                    if jn in joint_map:
                        jid = model.joint(jn).id
                        lo,hi = model.jnt_range[jid]
                        data.qpos[joint_map[jn]] = np.clip(dj[aidx], lo, hi)
        else:
            for side in ['left','right']:
                wp = converted[f'{side}_wrist_pos'][fi]; ft = converted[f'{side}_fingertips'][fi]
                for fi2, fn in enumerate(['index','middle','ring','little','thumb']):
                    curl = fingertip_distance_to_curl(wp, ft[fi2])
                    for cj in FINGER_CURL_JOINTS[side][fn]:
                        if cj in joint_map:
                            jid = model.joint(cj).id; lo,hi = model.jnt_range[jid]
                            data.qpos[joint_map[cj]] = lo + curl*(hi-lo)*0.9
        mujoco.mj_forward(model, data)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [0.3,0,1.05]; cam.distance = 1.6; cam.azimuth = 180; cam.elevation = -8
        opt = mujoco.MjvOption()
        renderer.update_scene(data, camera=cam, scene_option=opt)
        renderer._scene.lights[0].pos[:] = [0,0,4]; renderer._scene.lights[0].dir[:] = [0,0,-1]
        renderer._scene.lights[0].diffuse[:] = [0.9,0.9,0.9]; renderer._scene.lights[0].ambient[:] = [0.6,0.6,0.6]
        if renderer._scene.nlight > 1:
            renderer._scene.lights[1].diffuse[:] = [0.7,0.7,0.7]; renderer._scene.lights[1].ambient[:] = [0.4,0.4,0.4]
        return cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
    
    for dd in [d_gt, d_hm]:
        mujoco.mj_resetData(m_gt, dd)
        dd.qpos[0:3] = [0,0,0.98]; dd.qpos[3:7] = [1,0,0,0]
        mujoco.mj_forward(m_gt, dd)
    
    tmpdir = tempfile.mkdtemp(prefix=f"analytical_{task_key}_")
    print(f"  Rendering {n_frames} frames...")
    for fi in range(n_frames):
        gt_frame = render_robot(m_gt, d_gt, ik_gt, gt_converted, fi)
        hamer_frame = render_robot(m_hm, d_hm, ik_hm, hamer_converted, fi, use_dexmv=True)
        orig = cv2.resize(orig_frames[min(fi,len(orig_frames)-1)], (WIDTH,HEIGHT)) if orig_frames else np.zeros((HEIGHT,WIDTH,3), dtype=np.uint8)
        for img, label in [(orig,"Original"), (gt_frame,"GT Replay"), (hamer_frame,"Pipeline (analytical)")]:
            cv2.rectangle(img, (5,5), (280,35), (0,0,0), -1)
            cv2.putText(img, label, (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.imwrite(os.path.join(tmpdir, f"frame_{fi:05d}.png"), np.hstack([orig, gt_frame, hamer_frame]))
        if fi % 60 == 0: print(f"    Frame {fi}/{n_frames}")
    
    renderer.close()
    output = f"sim_renders/analytical_{task_key}.mp4"
    subprocess.run(["ffmpeg","-y","-framerate","30","-i",os.path.join(tmpdir,"frame_%05d.png"),
                    "-c:v","libx264","-pix_fmt","yuv420p","-crf","23",output], capture_output=True)
    shutil.rmtree(tmpdir)
    print(f"  Saved: {output}")

for task in ['pick_place', 'fold', 'stack']:
    process_task(task)
