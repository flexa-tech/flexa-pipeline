"""
Phone Hand Tracker — Replace a $350 data glove with an iPhone camera.

Uses MediaPipe Hands to extract 21 hand landmarks from RGB video,
computes joint angles (MCP/PIP/DIP per finger), wrist pose, and
grasp state. Outputs a JSON trajectory compatible with the pipeline.

Usage:
    python phone_hand_tracker.py input_video.mp4 -o output.json
    python phone_hand_tracker.py input_video.mp4 --hand right --preview
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

# ── MediaPipe landmark indices ──────────────────────────────────────
# https://google.github.io/mediapipe/solutions/hands.html
WRIST = 0
FINGER_LANDMARKS = {
    "thumb":  {"CMC": 1, "MCP": 2, "IP": 3, "TIP": 4},
    "index":  {"MCP": 5, "PIP": 6, "DIP": 7, "TIP": 8},
    "middle": {"MCP": 9, "PIP": 10, "DIP": 11, "TIP": 12},
    "ring":   {"MCP": 13, "PIP": 14, "DIP": 15, "TIP": 16},
    "pinky":  {"MCP": 17, "PIP": 18, "DIP": 19, "TIP": 20},
}

# Curl thresholds (radians) — finger is "curled" when angle < threshold
CURL_THRESHOLD = 2.0  # ~115°, fingers mostly straight > this
GRASP_MIN_CURLED = 3  # at least 3 fingers curled = grasping


def _vec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return b - a


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle in radians between two 3D vectors."""
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


def _landmark_to_xyz(lm, w: int, h: int) -> np.ndarray:
    """Convert a MediaPipe landmark to numpy [x, y, z] in pixel coords.
    z is relative depth from MediaPipe (unitless but proportional)."""
    return np.array([lm.x * w, lm.y * h, lm.z * w])  # z scaled by width


def compute_joint_angles(landmarks, w: int, h: int) -> dict:
    """Compute MCP, PIP, DIP angles for each finger from landmarks."""
    pts = {i: _landmark_to_xyz(landmarks[i], w, h) for i in range(21)}
    angles = {}

    for finger, ids in FINGER_LANDMARKS.items():
        if finger == "thumb":
            # Thumb: CMC-MCP angle, MCP-IP angle
            v1 = _vec(pts[ids["CMC"]], pts[WRIST])
            v2 = _vec(pts[ids["CMC"]], pts[ids["MCP"]])
            mcp_angle = _angle_between(v1, v2)

            v1 = _vec(pts[ids["MCP"]], pts[ids["CMC"]])
            v2 = _vec(pts[ids["MCP"]], pts[ids["IP"]])
            pip_angle = _angle_between(v1, v2)

            v1 = _vec(pts[ids["IP"]], pts[ids["MCP"]])
            v2 = _vec(pts[ids["IP"]], pts[ids["TIP"]])
            dip_angle = _angle_between(v1, v2)
        else:
            # Other fingers: standard MCP-PIP-DIP chain
            v1 = _vec(pts[ids["MCP"]], pts[WRIST])
            v2 = _vec(pts[ids["MCP"]], pts[ids["PIP"]])
            mcp_angle = _angle_between(v1, v2)

            v1 = _vec(pts[ids["PIP"]], pts[ids["MCP"]])
            v2 = _vec(pts[ids["PIP"]], pts[ids["DIP"]])
            pip_angle = _angle_between(v1, v2)

            v1 = _vec(pts[ids["DIP"]], pts[ids["PIP"]])
            v2 = _vec(pts[ids["DIP"]], pts[ids["TIP"]])
            dip_angle = _angle_between(v1, v2)

        angles[finger] = {
            "MCP": round(math.degrees(mcp_angle), 2),
            "PIP": round(math.degrees(pip_angle), 2),
            "DIP": round(math.degrees(dip_angle), 2),
        }

    return angles


def compute_finger_states(angles: dict) -> dict:
    """Return per-finger curl state: 'extended' or 'curled'."""
    states = {}
    for finger, a in angles.items():
        # A finger is curled if PIP angle is acute (bent)
        curled = a["PIP"] < math.degrees(CURL_THRESHOLD)
        states[finger] = "curled" if curled else "extended"
    return states


def detect_grasp(finger_states: dict) -> bool:
    curled_count = sum(1 for s in finger_states.values() if s == "curled")
    return curled_count >= GRASP_MIN_CURLED


def compute_wrist_pose(landmarks, w: int, h: int) -> dict:
    """Approximate wrist position and orientation."""
    wrist = _landmark_to_xyz(landmarks[WRIST], w, h)
    middle_mcp = _landmark_to_xyz(landmarks[9], w, h)
    index_mcp = _landmark_to_xyz(landmarks[5], w, h)

    # Palm normal (cross product of two palm vectors)
    v1 = _vec(wrist, middle_mcp)
    v2 = _vec(wrist, index_mcp)
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    if norm > 1e-8:
        normal = normal / norm

    return {
        "position": {"x": round(float(wrist[0]), 2),
                      "y": round(float(wrist[1]), 2),
                      "z": round(float(wrist[2]), 2)},
        "palm_normal": {"x": round(float(normal[0]), 4),
                        "y": round(float(normal[1]), 4),
                        "z": round(float(normal[2]), 4)},
    }


def process_video(
    video_path: str,
    output_path: Optional[str] = None,
    hand_filter: Optional[str] = None,  # "left", "right", or None (both)
    preview: bool = False,
    max_hands: int = 2,
    confidence: float = 0.5,
) -> dict:
    """Process a video file and return trajectory data."""

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video '{video_path}'", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {video_path} | {w}x{h} @ {fps:.1f} fps | {total} frames")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=max_hands,
        min_detection_confidence=confidence,
        min_tracking_confidence=confidence,
    )
    mp_draw = mp.solutions.drawing_utils if preview else None

    trajectory = {
        "metadata": {
            "source": "phone_hand_tracker",
            "video": str(Path(video_path).name),
            "fps": fps,
            "resolution": [w, h],
            "total_frames": total,
        },
        "frames": [],
    }

    frame_idx = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            frame_data = {
                "frame": frame_idx,
                "timestamp": round(frame_idx / fps, 4),
                "hands": [],
            }

            if results.multi_hand_landmarks:
                for hand_lm, hand_info in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness,
                ):
                    # MediaPipe labels are mirrored for selfie cam
                    label = hand_info.classification[0].label.lower()
                    score = round(hand_info.classification[0].score, 3)

                    if hand_filter and label != hand_filter:
                        continue

                    angles = compute_joint_angles(hand_lm.landmark, w, h)
                    finger_states = compute_finger_states(angles)
                    grasping = detect_grasp(finger_states)
                    wrist = compute_wrist_pose(hand_lm.landmark, w, h)

                    # Raw 3D landmarks for downstream use
                    landmarks_3d = []
                    for lm in hand_lm.landmark:
                        landmarks_3d.append({
                            "x": round(lm.x * w, 2),
                            "y": round(lm.y * h, 2),
                            "z": round(lm.z * w, 2),
                        })

                    hand_data = {
                        "hand": label,
                        "confidence": score,
                        "wrist": wrist,
                        "joint_angles": angles,
                        "finger_states": finger_states,
                        "grasping": grasping,
                        "landmarks_3d": landmarks_3d,
                    }
                    frame_data["hands"].append(hand_data)

                    if preview and mp_draw:
                        mp_draw.draw_landmarks(
                            frame, hand_lm, mp_hands.HAND_CONNECTIONS)

            trajectory["frames"].append(frame_data)

            if preview:
                # Draw grasp indicator
                for hd in frame_data["hands"]:
                    color = (0, 0, 255) if hd["grasping"] else (0, 255, 0)
                    txt = f"{hd['hand']} {'GRASP' if hd['grasping'] else 'open'}"
                    cv2.putText(frame, txt, (10, 30 + 30 * frame_data["hands"].index(hd)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.imshow("Hand Tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"  Processed {frame_idx}/{total} frames...")

    finally:
        cap.release()
        if preview:
            cv2.destroyAllWindows()
        hands.close()

    print(f"Done. {frame_idx} frames processed, "
          f"{sum(1 for f in trajectory['frames'] if f['hands'])} with hand detections.")

    # Save output
    if output_path is None:
        output_path = str(Path(video_path).with_suffix(".hand_trajectory.json"))

    with open(output_path, "w") as f:
        json.dump(trajectory, f, indent=2)
    print(f"Saved trajectory to {output_path}")

    return trajectory


def main():
    parser = argparse.ArgumentParser(
        description="Extract hand joint data from phone video — no glove needed.")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("-o", "--output", help="Output JSON path")
    parser.add_argument("--hand", choices=["left", "right"],
                        help="Track only one hand")
    parser.add_argument("--preview", action="store_true",
                        help="Show live preview window")
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Detection confidence threshold (0-1)")
    args = parser.parse_args()

    process_video(
        video_path=args.video,
        output_path=args.output,
        hand_filter=args.hand,
        preview=args.preview,
        max_hands=args.max_hands,
        confidence=args.confidence,
    )


if __name__ == "__main__":
    main()
