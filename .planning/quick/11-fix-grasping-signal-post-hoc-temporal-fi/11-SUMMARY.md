# QT-011 Summary: Fix grasping signal — post-hoc temporal filter + tighten HaMeR threshold

## Status: COMPLETE

## What changed

### Task 1: `clean_grasping_signal()` in `trim_trajectory.py` (0218d75)
- Added temporal filter that debounces short grasping bursts (<5 consecutive frames)
- Finds the first sustained grasp onset (5+ consecutive True frames)
- Clears all pre-onset noise to create a clean False->True transition
- Called in both `detect_action_window()` and `trim_calibrated_data()` — ensures clean signal regardless of entry point
- Function is idempotent (safe to call twice when detect_action_window is called from trim_calibrated_data)

### Task 2: HaMeR threshold tightened (33b9568)
- `_estimate_grasping_from_joints()` threshold: 0.04m (4cm) -> 0.02m (2cm)
- Docstring updated to reflect 2cm
- Takes effect on next Modal HaMeR run (doesn't affect existing cached data)

### Task 3: Diagnostic logging (included in 0218d75)
- `clean_grasping_signal()` prints before/after stats on every call
- Format: `GRASP CLEAN: {orig_count}/{n} ({pct}%) -> {clean_count}/{n} ({pct}%), onset frame={idx}`

## Files modified
- `trim_trajectory.py` — new `clean_grasping_signal()` function + integration calls
- `processing/hamer_modal.py` — threshold constant 0.04 -> 0.02

## Next step
Re-run pipeline with `--trim` to verify the cleaned grasping signal produces a proper approach->grasp transition in the trimmed window.
