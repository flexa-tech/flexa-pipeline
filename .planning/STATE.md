## Current Position

Phase: 1 — Foundations
Plan: 01 — Trajectory Smoothing (COMPLETE)
Status: Plan 01 complete, ready for plan 02
Last activity: 2026-03-14 — Plan 01 executed (TRK-01, TRK-02, TRK-03)

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-14)

**Core value:** The robot must faithfully reproduce the human's actual hand motion — true retargeting, not choreographed animation.
**Current focus:** Phase 1 — Foundations (trajectory smoothing complete, video trimming next)

## Accumulated Context

- v0.2 pipeline works end-to-end with real R3D data (stack2.r3d)
- STACKED=True passes for both synthetic and real data
- Key quality issues: choreographed grasp motion, 52% hand detection, 95s video length, trajectory jitter
- Branch: fix/e2e-pipeline (now with smoothing improvements)
- v0.3 roadmap created with 3 phases covering 14 requirements
- Phase 1 plan 01 complete: trajectory smoothing pipeline replaced
  - Bidirectional EMA → Savitzky-Golay (window=7, polyorder=3)
  - Linear np.interp → gap-length-aware (CubicSpline/PCHIP/linear)
  - Velocity clamp added (3cm/frame max)
  - Grasping signal guarded with binary assertion
  - STACKED=True regression test passes
- Phase 1 plan 02 (video trimming) is next

## Decisions Log

1. **Velocity clamp necessary** — 48% missing frames cause unavoidable interpolation artifacts; 3cm/frame clamp prevents teleportation
2. **SavGol window=7** — conservative, preserves grasp dwell positions
3. **Grasping never smoothed** — binary assertion guards against corruption
