# Flexa Pipeline

## What This Is

A motion retargeting pipeline that converts egocentric R3D video captures of human hand manipulation tasks into humanoid robot (Unitree G1) simulations in MuJoCo. Takes raw R3D recordings of block stacking, processes hand tracking and object detection, reconstructs 3D wrist trajectories, and drives a full-body G1 robot simulation via inverse kinematics.

## Core Value

The robot must faithfully reproduce the human's actual hand motion — true retargeting, not choreographed animation.

## Requirements

### Validated

<!-- Shipped and confirmed valuable. -->

- ✓ R3D ingest with ARKit pose extraction — v0.2
- ✓ Hand tracking via MediaPipe (fallback) — v0.2
- ✓ Object detection for block positions — v0.2
- ✓ 3D wrist reconstruction from egocentric video + ARKit — v0.2
- ✓ Workspace calibration (camera → robot coordinates) — v0.2
- ✓ G1 MuJoCo simulation with IK-driven arm motion — v0.2
- ✓ Kinematic block attachment (STACKED=True) — v0.2
- ✓ Pipeline stage orchestration with dependency tracking — v0.2
- ✓ Synthetic and real R3D data end-to-end — v0.2

### Active

<!-- Current scope. Building toward these. -->

- [ ] True wrist retargeting — real trajectory drives entire simulation, not just pre-grasp
- [ ] HaMeR hand pose estimation (>85% detection rate)
- [ ] Trajectory smoothing with gap interpolation
- [ ] Video trimming to action window (15-30s output)
- [ ] Improved grasp visual quality (finger pre-shaping, reduced interpenetration)

### Out of Scope

<!-- Explicit boundaries. Includes reasoning to prevent re-adding. -->

- Multi-hand manipulation — single dominant hand sufficient for block stacking
- Real-time inference — batch processing is fine for research pipeline
- Non-G1 robots — G1 is the target platform
- Dynamic grasping physics — kinematic attachment is acceptable for now

## Current Milestone: v0.3 True Retargeting & Quality

**Goal:** Replace choreographed motion with true human trajectory retargeting and improve visual output quality.

**Target features:**
- True wrist retargeting (real trajectory for entire sim)
- HaMeR integration (>85% hand detection)
- Trajectory smoothing (interpolate gaps, temporal filter)
- Video trimming (auto-detect action window)
- Grasp visual quality (finger pre-shaping)

## Context

- Pipeline processes R3D files: ingest → hand tracking → object detection → 3D reconstruction → calibration → simulation
- v0.2 uses a hardcoded progress variable `p` for grasp/carry/place — ignores human motion during grasping
- MediaPipe gets 52% detection on egocentric video; HaMeR stub raises NotImplementedError
- 48% frame gaps cause jerky arm motion — no smoothing applied
- 951-frame R3D renders as 95-second video; action starts at frame 577
- MuJoCo Menagerie G1 model at /Users/christian/Documents/ai_dev/mujoco_menagerie/unitree_g1/
- GPU stages (HaMeR) may use Modal cloud deployment

## Constraints

- **Platform**: MuJoCo + Unitree G1 via mujoco_menagerie
- **Data format**: Apple R3D files with ARKit pose data
- **GPU**: HaMeR inference requires GPU (Modal or local torch)
- **Python**: 3.11+ with .venv at project root

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Kinematic block attachment over physics grasp | Physics grasp too complex for current stage; kinematic gives reliable STACKED=True | ✓ Good |
| MediaPipe fallback when HaMeR unavailable | Ensures pipeline works without GPU, even at lower quality | ✓ Good |
| MuJoCo over Isaac Sim | Lighter weight, easier to iterate, sufficient for retargeting research | ✓ Good |
| Subsampled frames (4:1) for R3D processing | 951 frames too many; 235 frames sufficient for trajectory | — Pending |

---
*Last updated: 2026-03-13 after v0.3 milestone start*
