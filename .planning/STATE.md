---
gsd_state_version: 1.0
milestone: v0.3
milestone_name: milestone
status: verifying
last_updated: "2026-03-14T12:30:00Z"
last_activity: 2026-03-14 — Completed quick task 3: End-to-end HaMeR pipeline test on stack2
progress:
  total_phases: 3
  completed_phases: 3
  total_plans: 5
  completed_plans: 5
---

## Current Position

Phase: 3 — Quality Uplift
Plan: 02 — Grasp Visual Quality (CODE COMPLETE, verification pending)
Status: Finger pre-shaping + collision exclusion implemented. Needs `python mujoco_g1_v10.py stack2` to verify STACKED=True (OUT-05).
Last activity: 2026-03-14 — Completed quick task 2: Build Pinocchio-based trajectory validator

## Quick Tasks

- [x] QT-001: Automate session progress capture to Obsidian (vault-capture.sh)
- [x] QT-002: Build Pinocchio-based trajectory validator
- [x] QT-003: End-to-end HaMeR pipeline test on stack2

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 1 | Automate session progress capture to Obsidian | 2026-03-14 | 6a04467 | [1-vault-capture](./quick/1-vault-capture/) |
| 2 | Build Pinocchio-based trajectory validator | 2026-03-14 | 73ff194 | [2-build-pinocchio-based-trajectory-validat](./quick/2-build-pinocchio-based-trajectory-validat/) |
| 3 | End-to-end HaMeR pipeline test on stack2 | 2026-03-14 | 98f60a5 | [3-end-to-end-hamer-pipeline-test-on-stack2](./quick/3-end-to-end-hamer-pipeline-test-on-stack2/) |

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-14)

**Core value:** The robot must faithfully reproduce the human's actual hand motion — true retargeting, not choreographed animation.
**Current focus:** Phase 3 plan 02 code complete (grasp quality). Next: run simulation to verify STACKED=True, then deploy HaMeR Modal function.

## Accumulated Context

- v0.2 pipeline works end-to-end with real R3D data (stack2.r3d)
- STACKED=True passes for both synthetic and real data
- Key quality issues: choreographed grasp motion, 52% hand detection, trajectory jitter
- Branch: fix/e2e-pipeline (with smoothing + trimming improvements)
- v0.3 roadmap created with 3 phases covering 14 requirements
- Phase 1 plan 01 complete: trajectory smoothing pipeline replaced
  - Bidirectional EMA -> Savitzky-Golay (window=7, polyorder=3)
  - Linear np.interp -> gap-length-aware (CubicSpline/PCHIP/linear)
  - Velocity clamp added (3cm/frame max)
  - Grasping signal guarded with binary assertion
  - STACKED=True regression test passes
- Phase 1 plan 02 complete: video trimming module added
  - trim_trajectory.py: action window auto-detection via velocity + grasping density
  - Stack2 trimmed from 95s -> 22.9s (229 frames, window [547, 776))
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
- Phase 3 plan 01 code complete: HaMeR integration
  - processing/hamer_modal.py: Combined GroundingDINO + HaMeR on Modal A10G
  - egocrowd/hand_pose.py: Stub replaced with Modal remote call
  - run_pipeline.py: wrist_3d_camera passthrough + --hamer/--no-hamer flags
  - reconstruct_wrist_3d.py: cam_to_world helper + HaMeR 3D path skips depth lookup
  - Deployed and tested on stack2: 99% detection rate (vs 52% MediaPipe baseline)
  - Running in gdino-only mode (HaMeR mesh recovery fails to load on Modal)
  - Modal API updated: Function.lookup → Function.from_name (modal v1.3.5)
  - Detection threshold lowered from 0.2 → 0.15 (confidence values cluster around 0.20-0.21)
  - Grasping heuristic improved: aspect-ratio based (was box-area based)
  - No wrist_3d_camera output yet (requires HaMeR model to load); falls back to depth unprojection
  - Pipeline completes all 7 stages with --hamer flag
  - Simulation renders successfully (robot grasps and moves block, STACKED=False)
- Phase 3 plan 02 code complete: grasp visual quality
  - FINGER_PRESHAPE constant + PRESHAPE_DIST_START/FULL thresholds
  - Distance-based finger pre-shaping with quadratic ease-in during approach
  - Collision exclusion (contype/conaffinity=0) during kinematic hold
  - Collision restored on release
  - STACKED=True verification pending (needs simulation run)

## Decisions Log

1. **Velocity clamp necessary** -- 48% missing frames cause unavoidable interpolation artifacts; 3cm/frame clamp prevents teleportation
2. **SavGol window=7** -- conservative, preserves grasp dwell positions
3. **Grasping never smoothed** -- binary assertion guards against corruption
4. **In-place JSON trim** -- trim overwrites calibrated JSON rather than creating a new file, keeping downstream stages unchanged
5. **Sliding-window density for tightening** -- when action_frames span exceeds 300 frames, find densest cluster via sliding window (plan's margin-reduction alone was insufficient for stack2's 534-frame grasp span)
6. **Trim is opt-in** -- --trim flag defaults to off for backward compatibility
7. **Z floor clamp at table+2cm** -- wrist Z values (0.30-0.46m) from calibration are below table height (0.78m); clamping to 0.80m prevents below-table IK targets but creates systematic Z offset contributing to RMS error
8. **2cm IK convergence threshold** -- logs failures where palm-to-target exceeds 2cm; 108/229 frames exceed this, mostly due to Z floor clamping
9. **Combined Modal function** -- single A10G runs both GroundingDINO + HaMeR to avoid double cold-start penalty
10. **HaMeR --no-deps install** -- prevents mmcv/detectron2 transitive conflicts; manually install needed deps (smplx, timm, einops)
11. **Graceful 3-tier fallback** -- HaMeR mesh -> GroundingDINO detection-only -> MediaPipe (CPU); no frame-level mixing between models
12. **GroundingDINO threshold 0.15** -- egocentric hand views produce low confidence (~0.20-0.21); lowering from 0.2 to 0.15 increases detection from 27% to 99%
