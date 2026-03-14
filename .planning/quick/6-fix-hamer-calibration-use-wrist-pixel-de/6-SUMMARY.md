# QT-006 Summary: Fix HaMeR Calibration — Use wrist_pixel Depth Unprojection

**Status:** complete
**Branch:** fix/e2e-pipeline
**Commit:** 8fde274

## What Changed

In `reconstruct_wrist_3d.py`, the HaMeR `wrist_3d_camera` path (lines 219-227) previously used crop-relative camera coords for world position via `cam_to_world()` and returned early with `continue`, skipping depth unprojection. This produced garbage world coordinates because HaMeR's coords are relative to a 256x256 crop, not the full image.

**Fix:** Replaced the early-return block with a `pass` so the code falls through to the existing `wrist_pixel` depth unprojection path. The HaMeR grasping signal (derived from finger-tip distance) is still used — it's extracted downstream in the depth path. Updated the log message to reflect the new behavior.

## Verification Results

**Before (crop-relative HaMeR coords):**
- Wrist X: [0.886, 1.041] — outside G1 workspace [0.10, 0.65]
- Total X range: 0.155m (2.6cm of real motion)

**After (depth unprojection via wrist_pixel):**
- Wrist X: [0.196, 0.546] — well within G1 workspace
- Total X range: 0.350m of real motion
- Y: [0.026, 0.247], Z: [0.687, 0.877]
- 941/951 frames reconstructed (10 failed, same as before)
- Grasping signal intact (632 grasp frames)

## Files Modified

- `reconstruct_wrist_3d.py` — Removed early return from HaMeR block, updated log message
