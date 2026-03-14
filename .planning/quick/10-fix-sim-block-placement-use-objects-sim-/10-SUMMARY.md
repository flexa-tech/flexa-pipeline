# QT-010: Fix sim block placement — use objects_sim positions

## Problem

Block placement in `mujoco_g1_v10.py` derived positions from wrist trajectory extremes (grip onset/offset), ignoring the calibrated `objects_sim` positions from `--objects`. This placed blocks at incorrect locations (e.g., XY=(0.353, 0.215) instead of the calibrated (0.5, 0.01)).

## Fix

- When `obj_xy` has 2+ entries (from `objects_sim`), use those positions directly as `desired_pick_xy` and `desired_support_xy`
- Fall back to wrist-trajectory derivation only when fewer than 2 objects are available
- Handle sim-space coordinates correctly: `objects_sim` values are already in sim-space (no `table_x_offset` round-trip needed)

## Verification

```
Block placement from objects_sim:
  Pick block at:    XY=(0.500, 0.010)
  Support block at: XY=(0.500, -0.010)
```

Blocks now placed at calibrated positions. STACKED=False persists — remaining bottleneck is IK tracking (RMS=0.107m) and grasping signal (all frames grip=True, no approach phase differentiation).

## Commit

- `c9ba311` fix: use objects_sim positions for block placement instead of wrist trajectory derivation
