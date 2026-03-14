# QT-013 Summary: Proximity-Gated Grasping

## Result

Commit: `c8cb2b2`

Added proximity gating to `clean_grasping_signal()` in `trim_trajectory.py`. Grasping is now only allowed when the wrist is within 15cm of any object in `objects_sim`.

## Changes

**`trim_trajectory.py`:**

1. Extended `clean_grasping_signal()` signature with `wrist_positions`, `object_positions`, `proximity_threshold=0.15`
2. Added Step 0 (proximity gate) before debounce: for each frame where grasping is True, compute min Euclidean distance from wrist to all objects; zero out if distance > threshold
3. Added `GRASP PROXIMITY` diagnostic log line showing before/after counts
4. Updated `trim_calibrated_data()` to pass `wrist_positions=np.array(wrist_sim)` and `object_positions=np.array(objects_sim)` from calibration data
5. `detect_action_window()` call left unchanged (no object position data available in that path)

## Pipeline ordering

Proximity gate runs first, then debounce cleans up fragmented runs, then onset detection finds the clean transition. This ensures the aspect-ratio heuristic's false positives (palm at 0.334m from block) are eliminated before temporal filtering.

## Verification needed

Run full pipeline (`python run_pipeline.py stack2 --hamer --trim`) to confirm grasping signal is drastically reduced and only fires near the block.
