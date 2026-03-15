# Phone Hand Tracker

**Replace a $350 data glove with your iPhone camera.**

Uses MediaPipe Hands to extract 21 hand landmarks from RGB video, then computes joint angles, wrist pose, finger states, and grasp detection — outputting a JSON trajectory file compatible with our pipeline.

## How It Works

1. **Input**: RGB video from any camera (iPhone, webcam, etc.)
2. **MediaPipe Hands**: Detects 21 hand landmarks per frame with ML inference
3. **Joint Angle Computation**: For each finger (thumb, index, middle, ring, pinky), computes MCP, PIP, and DIP angles using 3D vector math on the landmark positions
4. **Depth Estimation**: MediaPipe provides a relative Z coordinate per landmark (monocular depth). This is approximate but usable for grasp detection and relative finger positioning
5. **Grasp Detection**: If 3+ fingers are curled (PIP angle below threshold), the hand is classified as grasping
6. **Output**: JSON with per-frame hand data matching our pipeline format

## Accuracy vs Physical Glove

| Aspect | Phone Camera | Data Glove |
|---|---|---|
| **Joint angles** | ±10-15° typical error | ±2-5° |
| **Depth (Z axis)** | Relative only, ~20% error | Absolute via IMU |
| **Finger contact** | Cannot detect | Pressure sensors |
| **Occlusion** | Fails when fingers hidden | No issue |
| **Frame rate** | 30fps typical | 100Hz+ |
| **Cost** | $0 (use your phone) | ~$350 |
| **Setup time** | 0 minutes | 5-10 min calibration |

**Bottom line**: Phone tracking gives ~80% of glove quality for manipulation tasks where fingers are visible. Good enough for training coarse grasp policies. Not sufficient for precision manipulation (threading needles, etc.).

## Usage

```bash
# Basic — process video, output JSON
python phone_hand_tracker.py capture.mp4

# Specify output path
python phone_hand_tracker.py capture.mp4 -o hand_data.json

# Track only right hand with preview window
python phone_hand_tracker.py capture.mp4 --hand right --preview

# Higher confidence threshold
python phone_hand_tracker.py capture.mp4 --confidence 0.7
```

### Requirements

```bash
pip install mediapipe opencv-python numpy
```

## Output Format

```json
{
  "metadata": {
    "source": "phone_hand_tracker",
    "video": "capture.mp4",
    "fps": 30.0,
    "resolution": [1920, 1080]
  },
  "frames": [
    {
      "frame": 0,
      "timestamp": 0.0,
      "hands": [
        {
          "hand": "right",
          "confidence": 0.95,
          "wrist": {
            "position": {"x": 500, "y": 300, "z": -20},
            "palm_normal": {"x": 0.0, "y": 0.0, "z": 1.0}
          },
          "joint_angles": {
            "thumb":  {"MCP": 45.0, "PIP": 30.0, "DIP": 15.0},
            "index":  {"MCP": 80.0, "PIP": 90.0, "DIP": 45.0},
            "middle": {"MCP": 85.0, "PIP": 95.0, "DIP": 50.0},
            "ring":   {"MCP": 80.0, "PIP": 90.0, "DIP": 45.0},
            "pinky":  {"MCP": 75.0, "PIP": 85.0, "DIP": 40.0}
          },
          "finger_states": {
            "thumb": "extended",
            "index": "curled",
            "middle": "curled",
            "ring": "curled",
            "pinky": "curled"
          },
          "grasping": true,
          "landmarks_3d": [{"x": 500, "y": 300, "z": -20}, "...21 total"]
        }
      ]
    }
  ]
}
```

## Pipeline Integration

The output JSON is designed to slot into our existing pipeline:

1. **Joint angles** → Direct input to robot joint mapping (after scaling to robot joint limits)
2. **Finger states** → Binary grasp signal for gripper control
3. **Wrist position** → Combined with body tracking (phone_mocap) for end-effector pose
4. **landmarks_3d** → Raw data for custom downstream processing

### With phone_mocap.py

Record hand and body simultaneously:
- Phone 1 (or front cam): Body tracking via phone_mocap.py
- Phone 2 (or selfie cam): Hand tracking via phone_hand_tracker.py
- Sync via timestamps, merge into unified trajectory

### Tips for Best Results

- **Lighting**: Bright, even lighting. Avoid harsh shadows on hands
- **Background**: Plain background helps detection
- **Distance**: Hand should fill ~30-50% of frame
- **Angle**: Palm facing camera gives best landmark accuracy
- **Stability**: Use a phone mount / tripod for consistent framing
- **Resolution**: 1080p is plenty; higher adds latency without accuracy gain
