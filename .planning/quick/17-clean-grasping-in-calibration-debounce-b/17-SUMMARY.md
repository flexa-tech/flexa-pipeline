# QT-017 Summary: Clean grasping in calibration — debounce before grasp centroid anchor

## What changed
- `calibrate_workspace.py`: Added `clean_grasping_signal()` call before computing the grasp centroid anchor
- Import added: `from trim_trajectory import clean_grasping_signal`
- The raw grasping signal (66% True = ~630 frames) is now debounced and onset-filtered before being used to compute the grasp centroid
- Cleaned grasping is automatically persisted in the calibrated JSON (existing `"grasping": grasping` at line 162)

## Why
The grasp centroid anchor aligns the wrist trajectory to detected object positions. With raw grasping (66% True), the centroid was essentially the trajectory center — not the actual grasp location. This caused systematic misalignment contributing to RMS=0.099m.

Debounce + onset detection removes early-trajectory false positives, so the centroid is computed from fewer, more accurate grasp frames closer to the actual object interaction.

## Limitations
- Proximity gate cannot be applied here because object positions are not yet aligned at this stage (that's what the centroid is computing)
- Debounce + onset still available from `clean_grasping_signal` which handles both

## Commit
- `058f835` — code change
