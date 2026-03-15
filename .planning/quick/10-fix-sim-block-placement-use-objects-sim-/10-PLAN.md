# QT-010: Fix sim block placement — use objects_sim positions

## Problem
Block placement in `mujoco_g1_v10.py` derives positions from wrist trajectory extremes, ignoring calibrated `objects_sim` positions from `--objects`.

## Task 1: Use objects_sim directly for block placement

**File:** mujoco_g1_v10.py

When `obj_xy` has 2+ entries (from `objects_sim`), use those positions directly as `desired_pick_xy` and `desired_support_xy`. Fall back to wrist-trajectory derivation only when fewer than 2 objects available.

**Acceptance:** Blocks placed at calibrated positions (0.500, 0.010) and (0.500, -0.010).
