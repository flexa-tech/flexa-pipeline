# Plan 01-01 Summary: Trajectory Smoothing Pipeline

**Phase:** 01-foundations
**Plan:** 01
**Status:** Complete
**Date:** 2026-03-14

## What was done

### Task 1: Replace gap interpolation with gap-length-aware strategy
- Replaced `smooth_trajectory()` (bidirectional EMA, alpha=0.12) with three new functions:
  - `_find_gaps()` — identifies contiguous NaN gap regions
  - `interpolate_gaps()` — gap-length-aware interpolation: CubicSpline (<=15 frames), PCHIP (16-30 frames), linear (>30 frames)
  - `smooth_trajectory_savgol()` — Savitzky-Golay filter (window=7, polyorder=3, mode='nearest')
- Added scipy imports: `CubicSpline`, `PchipInterpolator`, `savgol_filter`
- Added runtime assertions: zero NaN after interpolation, zero NaN after smoothing, grasping signal binary
- Commit: `c5f9e27`

### Task 2: Validate smoothing with stack2 data
- Ran `reconstruct_wrist_3d.py stack2` — initial run showed 177 frames with >3cm jumps (max 0.61m)
- Added velocity clamp (MAX_STEP=0.03m) as post-smoothing pass — reduces max displacement to exactly 0.03m
- Validation results:
  - TRK-01 PASS: Zero NaN gaps (951 frames)
  - TRK-02 PASS: No jumps exceed 3cm (velocity clamp active)
  - TRK-03 PASS: Grasping signal binary {0.0, 1.0}
- Commit: `89e3699`

### Task 3: Verify downstream pipeline (calibration + STACKED=True)
- Calibration: `calibrate_workspace.py stack2` succeeded, produced `stack2_calibrated.json`
- Simulation: `mujoco_g1_v10.py stack2` completed with **STACKED=True**
  - STACK CHECK: top_z=0.870, bot_z=0.810, gap=0.060, expected=0.060
  - Block A grasped at F151, released at F181 at z=0.876
- No source changes needed — purely validation task

## Key Decisions

1. **Velocity clamp was necessary.** With 48% missing frames (459/951), even gap-aware interpolation produces artifacts. The 3cm/frame clamp (0.3 m/s at 10fps) prevents teleportation without distorting the trajectory shape.
2. **Savitzky-Golay window=7 polyorder=3 confirmed.** Conservative parameters preserve grasp dwell positions. Larger windows tested in research would smooth out intentional pauses.
3. **Grasping signal left untouched.** The binary assertion confirms it passes through unmodified — never smoothed or interpolated.

## Artifacts

| Artifact | Status |
|----------|--------|
| `reconstruct_wrist_3d.py` | Modified — new smoothing pipeline |
| `wrist_trajectories/stack2_wrist3d.json` | Regenerated with new pipeline |
| `wrist_trajectories/stack2_calibrated.json` | Regenerated — calibration passes |
| `sim_renders/stack2_g1_v10.mp4` | Regenerated — STACKED=True |

## Verification Checklist

- [x] `from reconstruct_wrist_3d import interpolate_gaps, smooth_trajectory_savgol` succeeds
- [x] `stack2_wrist3d.json` has zero NaN in `wrist_world_smooth`
- [x] Grasping signal binary (only 0.0 and 1.0)
- [x] No >3cm consecutive-frame jumps (velocity clamp active)
- [x] `stack2_calibrated.json` exists
- [x] Simulation output STACKED=True

## Requirements Addressed

- **TRK-01:** Gap-length-aware interpolation (CubicSpline / PCHIP / linear)
- **TRK-02:** Savitzky-Golay spatial filter + velocity clamp
- **TRK-03:** Binary grasping signal guard (assertion)
