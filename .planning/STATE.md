## Current Position

Phase: 1 — Foundations
Plan: 02 — Video Trimming (COMPLETE)
Status: Phase 1 complete, ready for Phase 2
Last activity: 2026-03-14 — Plan 02 executed (OUT-01, OUT-02)

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-14)

**Core value:** The robot must faithfully reproduce the human's actual hand motion — true retargeting, not choreographed animation.
**Current focus:** Phase 1 complete — ready for Phase 2 (Core Retargeting)

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
- Phase 2 (Core Retargeting) is next

## Decisions Log

1. **Velocity clamp necessary** — 48% missing frames cause unavoidable interpolation artifacts; 3cm/frame clamp prevents teleportation
2. **SavGol window=7** — conservative, preserves grasp dwell positions
3. **Grasping never smoothed** — binary assertion guards against corruption
4. **In-place JSON trim** — trim overwrites calibrated JSON rather than creating a new file, keeping downstream stages unchanged
5. **Sliding-window density for tightening** — when action_frames span exceeds 300 frames, find densest cluster via sliding window (plan's margin-reduction alone was insufficient for stack2's 534-frame grasp span)
6. **Trim is opt-in** — --trim flag defaults to off for backward compatibility
