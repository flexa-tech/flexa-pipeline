# QT-006: Fix HaMeR Calibration — Use wrist_pixel Depth Unprojection

**Status:** complete
**Created:** 2026-03-14
**Estimated context:** ~20%

## Problem

HaMeR's `wrist_3d_camera` (from `pred_keypoints_3d` joint 0) is crop-relative — the wrist barely moves (2.6cm total range) because it's always near the center of the 256x256 cropped hand image. After `cam_to_world()`, these coords map to X=[0.886, 1.041] — far outside the G1 workspace [0.10, 0.65]. Meanwhile `wrist_pixel` (GroundingDINO box center) shows 768x570px of real motion.

## Root Cause

In `reconstruct_wrist_3d.py` lines 219-227, when HaMeR data has `wrist_3d_camera`, it uses that for world position via `cam_to_world()` and returns early, skipping depth unprojection. The crop-relative coords produce garbage world coordinates.

## Tasks

### Task 1: Modify HaMeR path to use depth unprojection for position (lines 219-227)

**File:** `reconstruct_wrist_3d.py`

**Changes:**
1. Remove the early `continue` from the `if t.get("wrist_3d_camera"):` block (lines 220-227)
2. Keep grasping extraction from HaMeR (the grasping signal derived from finger-tip distance IS valid)
3. Let the code fall through to the `wrist_pixel` depth unprojection path below (lines 229+)
4. Update the console log message (line 201) to reflect the new behavior

**Specifically, transform lines 219-227 from:**
```python
# TRK-05: Use HaMeR 3D wrist when available (skip depth lookup)
if t.get("wrist_3d_camera"):
    point_cam = np.array(t["wrist_3d_camera"])
    pose_idx = min(r3d_idx, len(meta["poses"]) - 1)
    pose = meta["poses"][pose_idx]
    point = cam_to_world(point_cam, pose)
    wrist_3d_world.append(point.tolist())
    grasping.append(t.get("grasping", False))
    continue
```

**To:**
```python
# TRK-05: HaMeR provides grasping signal; wrist position uses depth unprojection
# (wrist_3d_camera is crop-relative and NOT useful for world position)
if t.get("wrist_3d_camera"):
    # Grasping flag from HaMeR finger-tip distance is valid — extract it
    # but do NOT use wrist_3d_camera for position; fall through to depth unproject
    pass
```

The code then naturally falls through to the `wrist_pixel` depth unprojection at line 229+, which will use R3D depth maps to get proper world coordinates.

### Task 2: Verify calibration output

**Commands:**
```bash
cd /Users/christian/Documents/ai_dev/flexa-pipeline && source .venv/bin/activate
python reconstruct_wrist_3d.py stack2_hamer
python calibrate_workspace.py stack2_hamer
```

**Success criteria:**
- Wrist X range should be within or near [0.10, 0.65] (G1 workspace), NOT [0.886, 1.041]
- World coordinate ranges should show meaningful motion (not 2.6cm total)
- Grasping signal still present and binary

## Verification Checklist

- [ ] `wrist_3d_camera` no longer used for world position
- [ ] HaMeR grasping signal still propagated correctly
- [ ] Wrist X within G1 workspace bounds after calibration
- [ ] No regressions in non-HaMeR (MediaPipe) path
