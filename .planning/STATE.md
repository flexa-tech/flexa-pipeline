---
gsd_state_version: 1.0
milestone: v0.3
milestone_name: milestone
status: completed
last_updated: "2026-03-14T05:14:58.478Z"
last_activity: 2026-03-14 — Phase 2 Plan 02 executed (RET-01, RET-02, RET-03, RET-04)
progress:
  total_phases: 3
  completed_phases: 2
  total_plans: 3
  completed_plans: 3
---

## Current Position

Phase: 2 — Core Retargeting
Plan: 02 — Wrist Retargeting (COMPLETE)
Status: Phase 2 complete, ready for Phase 3
Last activity: 2026-03-14 — Phase 2 Plan 02 executed (RET-01, RET-02, RET-03, RET-04)

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-14)

**Core value:** The robot must faithfully reproduce the human's actual hand motion — true retargeting, not choreographed animation.
**Current focus:** Phase 2 complete — ready for Phase 3 (Quality Uplift)

## Accumulated Context

- v0.2 pipeline works end-to-end with real R3D data (stack2.r3d)
- STACKED=True passes for both synthetic and real data
- Key quality issues: choreographed grasp motion, 52% hand detection, trajectory jitter
- Branch: fix/e2e-pipeline (with smoothing + trimming improvements)
- v0.3 roadmap created with 3 phases covering 14 requirements
- Phase 1 plan 01 complete: trajectory smoothing pipeline replaced
  - Bidirectional EMA → Savitzky-Golay (window=7, polyorder=3)
  - Linear np.interp → gap-length-aware (CubicSpline/PCHIP/linear)
  - Velocity clamp added (3cm/frame max)
  - Grasping signal guarded with binary assertion
  - STACKED=True regression test passes
- Phase 1 plan 02 complete: video trimming module added
  - trim_trajectory.py: action window auto-detection via velocity + grasping density
  - Stack2 trimmed from 95s → 22.9s (229 frames, window [547, 776))
  - Sliding-window cluster focus enforces 15-30s duration target
  - Pipeline has --trim, --trim-start, --trim-end CLI flags (stage 6, total now 7)
  - trim_info metadata in calibrated JSON for traceability
  - STACKED=True regression test passes with trimmed data
- Phase 2 plan 02 complete: core retargeting implemented
  - `p` progress variable completely eliminated from frame loop
  - IK target for ALL frames comes from `wrist[i]` (real human trajectory)
  - `want_grip` derives from `grasping[i]` (tracking data), not p thresholds
  - Workspace clamping via WORKSPACE_MIN/WORKSPACE_MAX constants
  - IK convergence logging with frame numbers and error magnitudes
  - RMS tracking error: 0.0555m (slightly above 0.05m threshold)
  - Systematic Z offset is root cause (wrist Z 0.30-0.46 clamped to 0.80 table height)
  - STACKED=True still passes (bonus -- not required for Phase 2)
- Phase 3 (Quality Uplift) is next

## Decisions Log

1. **Velocity clamp necessary** — 48% missing frames cause unavoidable interpolation artifacts; 3cm/frame clamp prevents teleportation
2. **SavGol window=7** — conservative, preserves grasp dwell positions
3. **Grasping never smoothed** — binary assertion guards against corruption
4. **In-place JSON trim** — trim overwrites calibrated JSON rather than creating a new file, keeping downstream stages unchanged
5. **Sliding-window density for tightening** — when action_frames span exceeds 300 frames, find densest cluster via sliding window (plan's margin-reduction alone was insufficient for stack2's 534-frame grasp span)
6. **Trim is opt-in** — --trim flag defaults to off for backward compatibility
7. **Z floor clamp at table+2cm** — wrist Z values (0.30-0.46m) from calibration are below table height (0.78m); clamping to 0.80m prevents below-table IK targets but creates systematic Z offset contributing to RMS error
8. **2cm IK convergence threshold** — logs failures where palm-to-target exceeds 2cm; 108/229 frames exceed this, mostly due to Z floor clamping
