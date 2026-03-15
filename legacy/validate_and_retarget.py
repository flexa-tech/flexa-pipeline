"""
Validate + Retarget + Render robot hand visualization.

1. Validation: overlay MediaPipe hand landmarks on RGB frames → video
2. Retarget: map human 21-joint angles → Allegro Hand 16-DoF
3. Render: MuJoCo visualization of retargeted robot hand motion → video
4. Accuracy: compute metrics (smoothness, coverage, joint range validity)
"""

import json
import os
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))


# ── Hand landmark connections for drawing ──
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),    # Thumb
    (0,5),(5,6),(6,7),(7,8),    # Index
    (0,9),(9,10),(10,11),(11,12),  # Middle
    (0,13),(13,14),(14,15),(15,16), # Ring
    (0,17),(17,18),(18,19),(19,20), # Pinky
    (5,9),(9,13),(13,17),        # Palm
]


def create_validation_video(session_dir: str, output_path: str, max_frames: int = 900) -> Dict:
    """Overlay hand landmarks on RGB frames and create validation video."""
    session_dir = Path(session_dir)
    
    # Load hand trajectory
    hand_path = session_dir / "hand_trajectory.json"
    if not hand_path.exists():
        return {"error": "no hand trajectory"}
    
    with open(hand_path) as f:
        traj = json.load(f)
    
    frames_dir = session_dir / "frames"
    if not frames_dir.exists():
        return {"error": "no frames dir"}
    
    resolution = traj.get("metadata", {}).get("resolution", [960, 720])
    w, h = resolution
    fps = traj.get("metadata", {}).get("fps", 60)
    # Output at 30fps
    out_fps = 30
    skip = max(1, int(fps / out_fps))
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, out_fps, (w, h))
    
    frames_data = traj.get("frames", [])
    written = 0
    grasp_frames = 0
    total_hands = 0
    
    for i in range(0, len(frames_data), skip):
        if written >= max_frames:
            break
        
        frame_info = frames_data[i]
        frame_path = frames_dir / f"{i:06d}.jpg"
        if not frame_path.exists():
            continue
        
        img = cv2.imread(str(frame_path))
        if img is None:
            continue
        
        if img.shape[1] != w or img.shape[0] != h:
            img = cv2.resize(img, (w, h))
        
        hands = frame_info.get("hands", [])
        
        for hand in hands:
            total_hands += 1
            landmarks = hand.get("landmarks_3d", [])
            grasping = hand.get("grasping", False)
            label = hand.get("hand", "?")
            confidence = hand.get("confidence", 0)
            
            if grasping:
                grasp_frames += 1
            
            # Draw landmarks
            if len(landmarks) == 21:
                pts = [(int(lm["x"]), int(lm["y"])) for lm in landmarks]
                
                # Draw connections
                color = (0, 0, 255) if grasping else (0, 255, 0)
                for c1, c2 in HAND_CONNECTIONS:
                    cv2.line(img, pts[c1], pts[c2], color, 2)
                
                # Draw points
                for j, pt in enumerate(pts):
                    r = 5 if j == 0 else 3  # Bigger for wrist
                    cv2.circle(img, pt, r, (255, 255, 255), -1)
                    cv2.circle(img, pt, r, color, 1)
                
                # Label
                txt = f"{label} {'GRASP' if grasping else 'open'} ({confidence:.0%})"
                cv2.putText(img, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        if not hands:
            cv2.putText(img, "NO HANDS DETECTED", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
        
        # Frame counter
        cv2.putText(img, f"Frame {i}/{len(frames_data)}", (w-200, h-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        writer.write(img)
        written += 1
    
    writer.release()
    
    size_mb = os.path.getsize(output_path) / (1024*1024)
    return {
        "frames_written": written,
        "total_hand_detections": total_hands,
        "grasp_frames": grasp_frames,
        "size_mb": round(size_mb, 1),
    }


# ── Retargeting: Human MediaPipe joints → Allegro Hand 16-DoF ──

# Allegro Hand has 16 joints: 4 per finger (index, middle, ring, thumb)
# Each finger: [J0_rotation, J1_flexion, J2_flexion, J3_flexion]
# Thumb is special: [J12_rotation, J13_flexion, J14_flexion, J15_flexion]

# MediaPipe gives us MCP, PIP, DIP angles per finger
# Mapping: MCP → J0+J1, PIP → J2, DIP → J3

ALLEGRO_JOINT_LIMITS = {
    # Index finger (joints 0-3)
    0: (0.263, 1.396),    # J0 rotation
    1: (-0.105, 1.163),   # J1 flexion
    2: (-0.189, 1.644),   # J2 flexion
    3: (-0.162, 1.644),   # J3 flexion
    # Middle finger (joints 4-7)
    4: (0.263, 1.396),
    5: (-0.105, 1.163),
    6: (-0.189, 1.644),
    7: (-0.162, 1.644),
    # Ring finger (joints 8-11) - note: ring = pinky in Allegro mapping
    8: (0.263, 1.396),
    9: (-0.105, 1.163),
    10: (-0.189, 1.644),
    11: (-0.162, 1.644),
    # Thumb (joints 12-15)
    12: (0.263, 1.396),   # Rotation
    13: (-0.105, 1.163),  # Flexion
    14: (-0.189, 1.644),
    15: (-0.162, 1.644),
}


def retarget_frame_to_allegro(joint_angles: Dict) -> List[float]:
    """
    Map MediaPipe joint angles (degrees) to Allegro Hand 16-DoF (radians).
    
    MediaPipe gives: MCP, PIP, DIP per finger (in degrees)
    Allegro expects: 4 joints per finger in radians, within joint limits
    """
    allegro = [0.0] * 16
    
    finger_map = {
        "index": 0,    # Allegro joints 0-3
        "middle": 4,   # Allegro joints 4-7
        "ring": 8,     # Allegro joints 8-11 (we map ring+pinky avg here)
        "thumb": 12,   # Allegro joints 12-15
    }
    
    for finger, base_idx in finger_map.items():
        if finger in joint_angles:
            angles = joint_angles[finger]
        elif finger == "ring" and "ring" in joint_angles:
            angles = joint_angles["ring"]
            # Average with pinky if available
            if "pinky" in joint_angles:
                pinky = joint_angles["pinky"]
                angles = {
                    k: (angles[k] + pinky[k]) / 2 for k in angles
                }
        else:
            continue
        
        mcp_deg = angles.get("MCP", 0)
        pip_deg = angles.get("PIP", 0)
        dip_deg = angles.get("DIP", 0)
        
        # Convert to radians
        mcp_rad = np.radians(mcp_deg)
        pip_rad = np.radians(pip_deg)
        dip_rad = np.radians(dip_deg)
        
        # Map to Allegro joints
        # J0: abduction/rotation — derive from MCP lateral component (small)
        j0_min, j0_max = ALLEGRO_JOINT_LIMITS[base_idx]
        # J1: MCP flexion
        j1_min, j1_max = ALLEGRO_JOINT_LIMITS[base_idx + 1]
        # J2: PIP flexion
        j2_min, j2_max = ALLEGRO_JOINT_LIMITS[base_idx + 2]
        # J3: DIP flexion
        j3_min, j3_max = ALLEGRO_JOINT_LIMITS[base_idx + 3]
        
        # Scale MediaPipe angles to Allegro joint ranges
        # MediaPipe MCP typically 0-180°, PIP 0-180°, DIP 0-180°
        # Allegro ranges are much smaller (~0.2 to 1.6 rad)
        
        allegro[base_idx] = np.clip(
            j0_min + (mcp_rad / np.pi) * (j0_max - j0_min) * 0.3,  # Small rotation
            j0_min, j0_max
        )
        allegro[base_idx + 1] = np.clip(
            j1_min + (mcp_rad / np.pi) * (j1_max - j1_min),
            j1_min, j1_max
        )
        allegro[base_idx + 2] = np.clip(
            j2_min + (pip_rad / np.pi) * (j2_max - j2_min),
            j2_min, j2_max
        )
        allegro[base_idx + 3] = np.clip(
            j3_min + (dip_rad / np.pi) * (j3_max - j3_min),
            j3_min, j3_max
        )
    
    return allegro


def retarget_session(session_dir: str) -> Dict:
    """Retarget all frames in a session to Allegro Hand."""
    session_dir = Path(session_dir)
    
    hand_path = session_dir / "hand_trajectory.json"
    if not hand_path.exists():
        return {"error": "no hand trajectory"}
    
    with open(hand_path) as f:
        traj = json.load(f)
    
    frames = traj.get("frames", [])
    retargeted = []
    valid_frames = 0
    
    for frame in frames:
        hands = frame.get("hands", [])
        if hands:
            # Use first/right hand
            hand = hands[0]
            for h in hands:
                if h.get("hand") == "right":
                    hand = h
                    break
            
            angles = hand.get("joint_angles", {})
            allegro_joints = retarget_frame_to_allegro(angles)
            retargeted.append({
                "timestamp": frame["timestamp"],
                "allegro_joints": allegro_joints,
                "grasping": hand.get("grasping", False),
                "has_hand": True,
            })
            valid_frames += 1
        else:
            # Hold last pose
            last_joints = retargeted[-1]["allegro_joints"] if retargeted else [0.0] * 16
            retargeted.append({
                "timestamp": frame["timestamp"],
                "allegro_joints": last_joints,
                "grasping": False,
                "has_hand": False,
            })
    
    # Save retargeted data
    out_path = session_dir / "allegro_retargeted.json"
    with open(out_path, "w") as f:
        json.dump({
            "target": "allegro_hand",
            "num_joints": 16,
            "joint_limits": {str(k): list(v) for k, v in ALLEGRO_JOINT_LIMITS.items()},
            "frames": retargeted,
            "valid_frames": valid_frames,
            "total_frames": len(frames),
        }, f)
    
    return {
        "valid_frames": valid_frames,
        "total_frames": len(frames),
        "coverage": round(valid_frames / max(len(frames), 1) * 100, 1),
    }


def render_robot_hand_video(session_dir: str, output_path: str, max_frames: int = 900) -> Dict:
    """
    Render a side-by-side video: original RGB | Allegro hand joint visualization.
    Uses a 2D schematic rendering (no MuJoCo needed for this step).
    """
    session_dir = Path(session_dir)
    
    retarget_path = session_dir / "allegro_retargeted.json"
    if not retarget_path.exists():
        return {"error": "no retargeted data"}
    
    with open(retarget_path) as f:
        retarget_data = json.load(f)
    
    hand_path = session_dir / "hand_trajectory.json"
    with open(hand_path) as f:
        traj = json.load(f)
    
    frames_dir = session_dir / "frames"
    resolution = traj.get("metadata", {}).get("resolution", [960, 720])
    w, h = resolution
    fps = traj.get("metadata", {}).get("fps", 60)
    out_fps = 30
    skip = max(1, int(fps / out_fps))
    
    # Output video: side by side (RGB + robot hand viz)
    viz_w = 400
    out_w = w + viz_w
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, out_fps, (out_w, h))
    
    retarget_frames = retarget_data.get("frames", [])
    written = 0
    
    for i in range(0, len(retarget_frames), skip):
        if written >= max_frames:
            break
        
        # Load RGB frame
        frame_path = frames_dir / f"{i:06d}.jpg"
        if frame_path.exists():
            img = cv2.imread(str(frame_path))
            if img is not None and (img.shape[1] != w or img.shape[0] != h):
                img = cv2.resize(img, (w, h))
        else:
            img = np.zeros((h, w, 3), dtype=np.uint8)
        
        if img is None:
            img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Draw hand landmarks on RGB
        if i < len(traj.get("frames", [])):
            hands = traj["frames"][i].get("hands", [])
            for hand in hands:
                landmarks = hand.get("landmarks_3d", [])
                grasping = hand.get("grasping", False)
                if len(landmarks) == 21:
                    pts = [(int(lm["x"]), int(lm["y"])) for lm in landmarks]
                    color = (0, 0, 255) if grasping else (0, 255, 0)
                    for c1, c2 in HAND_CONNECTIONS:
                        cv2.line(img, pts[c1], pts[c2], color, 2)
                    for pt in pts:
                        cv2.circle(img, pt, 3, (255, 255, 255), -1)
        
        # Create robot hand visualization panel
        viz = np.zeros((h, viz_w, 3), dtype=np.uint8)
        viz[:] = (30, 30, 30)  # Dark background
        
        rf = retarget_frames[i] if i < len(retarget_frames) else {"allegro_joints": [0]*16, "has_hand": False}
        joints = rf["allegro_joints"]
        has_hand = rf.get("has_hand", False)
        grasping = rf.get("grasping", False)
        
        # Title
        status_color = (0, 255, 0) if has_hand else (0, 0, 200)
        cv2.putText(viz, "Allegro Hand", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        status = "GRASP" if grasping else ("TRACKING" if has_hand else "NO HAND")
        cv2.putText(viz, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Draw simplified hand schematic
        # Palm center
        cx, cy = viz_w // 2, h // 2 + 50
        palm_w, palm_h = 80, 100
        
        # Palm rectangle
        palm_color = (0, 200, 0) if has_hand else (80, 80, 80)
        cv2.rectangle(viz, (cx - palm_w//2, cy - palm_h//2), 
                      (cx + palm_w//2, cy + palm_h//2), palm_color, 2)
        
        # Draw each finger as articulated segments
        finger_names = ["Index", "Middle", "Ring", "Thumb"]
        finger_bases = [
            (cx - 30, cy - palm_h//2),   # Index
            (cx, cy - palm_h//2),         # Middle
            (cx + 30, cy - palm_h//2),    # Ring
            (cx - palm_w//2 - 10, cy - 20),  # Thumb (side)
        ]
        
        for fi, (fname, base) in enumerate(zip(finger_names, finger_bases)):
            base_joint = fi * 4
            seg_len = 30
            
            # Get joint angles (radians)
            j0 = joints[base_joint] if base_joint < len(joints) else 0
            j1 = joints[base_joint+1] if base_joint+1 < len(joints) else 0
            j2 = joints[base_joint+2] if base_joint+2 < len(joints) else 0
            j3 = joints[base_joint+3] if base_joint+3 < len(joints) else 0
            
            # Draw articulated chain
            if fi == 3:  # Thumb goes sideways
                angle = -np.pi/2 - j0 * 0.5
            else:
                angle = -np.pi/2 + j0 * 0.3  # Slight splay
            
            p0 = base
            segments = [j1, j2, j3]
            prev = p0
            
            for si, jangle in enumerate(segments):
                angle -= jangle * 0.6  # Curl
                next_x = int(prev[0] + seg_len * np.cos(angle))
                next_y = int(prev[1] + seg_len * np.sin(angle))
                next_pt = (next_x, next_y)
                
                seg_color = (0, 200, 255) if has_hand else (100, 100, 100)
                cv2.line(viz, prev, next_pt, seg_color, 3)
                cv2.circle(viz, prev, 4, (255, 255, 255), -1)
                prev = next_pt
            
            cv2.circle(viz, prev, 4, (0, 0, 255), -1)  # Fingertip
            
            # Label
            cv2.putText(viz, fname, (base[0]-15, base[1]+palm_h+30+fi*18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
        
        # Joint values display
        y_offset = h - 200
        cv2.putText(viz, "Joint Values (rad):", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        for fi, fname in enumerate(["Idx", "Mid", "Rng", "Thm"]):
            base_j = fi * 4
            vals = joints[base_j:base_j+4]
            txt = f"{fname}: " + " ".join(f"{v:.2f}" for v in vals)
            cv2.putText(viz, txt, (10, y_offset + 20 + fi * 18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
        
        # Combine
        combined = np.hstack([img, viz])
        writer.write(combined)
        written += 1
    
    writer.release()
    size_mb = os.path.getsize(output_path) / (1024*1024)
    return {"frames": written, "size_mb": round(size_mb, 1)}


def compute_accuracy_metrics(session_dir: str) -> Dict:
    """Compute pipeline accuracy metrics for a session."""
    session_dir = Path(session_dir)
    
    retarget_path = session_dir / "allegro_retargeted.json"
    hand_path = session_dir / "hand_trajectory.json"
    
    if not retarget_path.exists() or not hand_path.exists():
        return {}
    
    with open(retarget_path) as f:
        retarget = json.load(f)
    with open(hand_path) as f:
        traj = json.load(f)
    
    frames = retarget.get("frames", [])
    if not frames:
        return {}
    
    # 1. Coverage: % frames with hand detection
    coverage = sum(1 for f in frames if f["has_hand"]) / len(frames)
    
    # 2. Joint range utilization: what % of Allegro joint range is used
    all_joints = np.array([f["allegro_joints"] for f in frames if f["has_hand"]])
    if len(all_joints) == 0:
        return {"coverage": round(coverage, 3)}
    
    joint_ranges_used = []
    for j in range(16):
        j_min, j_max = ALLEGRO_JOINT_LIMITS[j]
        j_range = j_max - j_min
        actual_range = all_joints[:, j].max() - all_joints[:, j].min()
        utilization = actual_range / j_range if j_range > 0 else 0
        joint_ranges_used.append(utilization)
    
    # 3. Smoothness: average jerk (lower = smoother)
    if len(all_joints) > 3:
        velocities = np.diff(all_joints, axis=0)
        accelerations = np.diff(velocities, axis=0)
        jerk = np.diff(accelerations, axis=0)
        smoothness = float(np.mean(np.abs(jerk)))
    else:
        smoothness = 0.0
    
    # 4. Grasp detection rate
    grasp_frames = sum(1 for f in frames if f.get("grasping", False))
    grasp_rate = grasp_frames / len(frames)
    
    # 5. Joint validity: % of joint values within Allegro limits
    valid_count = 0
    total_count = 0
    for f in frames:
        if f["has_hand"]:
            for j, val in enumerate(f["allegro_joints"]):
                j_min, j_max = ALLEGRO_JOINT_LIMITS[j]
                if j_min <= val <= j_max:
                    valid_count += 1
                total_count += 1
    
    joint_validity = valid_count / max(total_count, 1)
    
    return {
        "coverage": round(coverage, 3),
        "avg_joint_range_utilization": round(float(np.mean(joint_ranges_used)), 3),
        "smoothness_jerk": round(smoothness, 6),
        "grasp_detection_rate": round(grasp_rate, 3),
        "joint_validity": round(joint_validity, 3),
        "num_valid_frames": int(sum(1 for f in frames if f["has_hand"])),
    }


def process_all(sessions_dir: str, output_dir: str):
    """Run validation, retargeting, and rendering on all sessions."""
    sessions_dir = Path(sessions_dir)
    output_dir = Path(output_dir)
    os.makedirs(str(output_dir), exist_ok=True)
    
    session_dirs = sorted([d for d in sessions_dir.iterdir() if d.is_dir()])
    
    print(f"Processing {len(session_dirs)} sessions")
    print("=" * 70)
    
    all_metrics = {}
    
    for sd in session_dirs:
        name = sd.name
        print(f"\n{'─'*50}")
        print(f"Session: {name}")
        
        # 1. Validation video
        val_path = str(output_dir / f"{name}_validation.mp4")
        print(f"  Creating validation video...")
        val_result = create_validation_video(str(sd), val_path)
        print(f"    → {val_result.get('frames_written', 0)} frames, {val_result.get('size_mb', 0)} MB")
        
        # 2. Retarget
        print(f"  Retargeting to Allegro Hand...")
        ret_result = retarget_session(str(sd))
        print(f"    → {ret_result.get('valid_frames', 0)}/{ret_result.get('total_frames', 0)} frames ({ret_result.get('coverage', 0)}%)")
        
        # 3. Robot hand visualization
        robot_path = str(output_dir / f"{name}_robot_hand.mp4")
        print(f"  Rendering robot hand video...")
        render_result = render_robot_hand_video(str(sd), robot_path)
        print(f"    → {render_result.get('frames', 0)} frames, {render_result.get('size_mb', 0)} MB")
        
        # 4. Accuracy metrics
        metrics = compute_accuracy_metrics(str(sd))
        all_metrics[name] = metrics
        print(f"  Metrics:")
        for k, v in metrics.items():
            print(f"    {k}: {v}")
    
    # Summary
    print(f"\n{'='*70}")
    print("PIPELINE ACCURACY REPORT")
    print(f"{'='*70}")
    
    coverages = [m["coverage"] for m in all_metrics.values() if "coverage" in m]
    validities = [m["joint_validity"] for m in all_metrics.values() if "joint_validity" in m]
    utilizations = [m["avg_joint_range_utilization"] for m in all_metrics.values() if "avg_joint_range_utilization" in m]
    smoothnesses = [m["smoothness_jerk"] for m in all_metrics.values() if "smoothness_jerk" in m]
    
    print(f"\nHand Detection Coverage: {np.mean(coverages)*100:.1f}% avg (range: {np.min(coverages)*100:.0f}%-{np.max(coverages)*100:.0f}%)")
    print(f"Joint Validity (within Allegro limits): {np.mean(validities)*100:.1f}%")
    print(f"Joint Range Utilization: {np.mean(utilizations)*100:.1f}%")
    print(f"Smoothness (avg jerk, lower=better): {np.mean(smoothnesses):.6f}")
    
    # Per-task summary
    by_task = defaultdict(list)
    for name, metrics in all_metrics.items():
        # Infer task from name
        if "pick" in name: task = "pick_place"
        elif "pour" in name: task = "pour"
        elif "stack" in name: task = "stack"
        elif "sort" in name: task = "sort"
        elif "fold" in name: task = "fold"
        elif "drawer" in name: task = "open_drawer"
        elif "assemble" in name: task = "assemble"
        else: task = "other"
        by_task[task].append(metrics)
    
    print(f"\nPer-Task Breakdown:")
    for task, metrics_list in sorted(by_task.items()):
        avg_cov = np.mean([m.get("coverage", 0) for m in metrics_list])
        avg_val = np.mean([m.get("joint_validity", 0) for m in metrics_list])
        avg_grasp = np.mean([m.get("grasp_detection_rate", 0) for m in metrics_list])
        print(f"  {task}: coverage={avg_cov*100:.0f}%, validity={avg_val*100:.0f}%, grasp_rate={avg_grasp*100:.0f}%")
    
    # Save report
    report = {
        "summary": {
            "num_sessions": len(all_metrics),
            "avg_coverage": round(float(np.mean(coverages)), 3),
            "avg_joint_validity": round(float(np.mean(validities)), 3),
            "avg_range_utilization": round(float(np.mean(utilizations)), 3),
            "avg_smoothness": round(float(np.mean(smoothnesses)), 6),
        },
        "per_session": all_metrics,
    }
    
    report_path = str(output_dir / "accuracy_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {report_path}")
    
    # List output videos
    print(f"\nOutput videos in {output_dir}:")
    for f in sorted(output_dir.glob("*.mp4")):
        size_mb = f.stat().st_size / (1024*1024)
        print(f"  {f.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", default="processed_sessions")
    parser.add_argument("--output", default="validation_output")
    args = parser.parse_args()
    
    process_all(args.sessions, args.output)
