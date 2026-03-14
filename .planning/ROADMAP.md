# Roadmap: Flexa Pipeline v0.3

## Overview

| Phase | Name | Goal | Requirements | Criteria |
|-------|------|------|-------------|----------|
| 1 | Foundations | Smooth trajectories and trim video to unblock retargeting and speed up iteration | TRK-01, TRK-02, TRK-03, OUT-01, OUT-02 | 5 |
| 2 | Core Retargeting | Replace choreographed motion with real human wrist trajectory driving the full simulation | RET-01, RET-02, RET-03, RET-04 | 4 |
| 3 | Quality Uplift | Upgrade hand detection with HaMeR and improve grasp visual fidelity | TRK-04, TRK-05, OUT-03, OUT-04, OUT-05 | 5 |

## Phase Details

### Phase 1: Foundations --- COMPLETE (2026-03-14)
**Goal:** Smooth raw trajectories and trim video output to make retargeting viable and iteration fast.
**Requirements:** TRK-01, TRK-02, TRK-03, OUT-01, OUT-02

Two parallel workstreams:

**1a — Trajectory Smoothing (TRK-01, TRK-02, TRK-03)** --- COMPLETE (Plan 01, 2026-03-14)
Interpolate gaps with gap-length-aware strategy and apply Savitzky-Golay spatial filter while preserving the binary grasping signal.
- Plan 01 complete: EMA replaced with SavGol, gap-aware interpolation, velocity clamp, grasping guard. STACKED=True passes.

**1b — Video Trimming (OUT-01, OUT-02)** --- COMPLETE (Plan 02, 2026-03-14)
Auto-detect action window from grasping signal and wrist velocity; cut 95s renders down to 15-30s.
- Plan 02 complete: trim_trajectory.py with velocity+grasping detection, sliding-window cluster focus. Stack2 trimmed 95s->22.9s. STACKED=True passes.

**Success Criteria:**
1. Smoothed wrist trajectory has zero NaN gaps and no jumps >3cm between consecutive frames
2. Grasping signal remains binary (0 or 1) after smoothing — never interpolated
3. Output video starts within 1 second of first grasp approach and ends within 2 seconds of last release
4. Output video duration is between 15 and 30 seconds for the test recording (stack2.r3d)
5. STACKED=True still passes with smoothed + trimmed data

### Phase 2: Core Retargeting
**Goal:** Eliminate the choreographed `p` progress variable and drive the robot arm from the real human wrist trajectory for every frame of the simulation.
**Requirements:** RET-01, RET-02, RET-03, RET-04

**Success Criteria:**
1. Robot wrist position tracks the human wrist trajectory with <5cm RMS error across all frames (within reachable workspace)
2. Grasp triggers from `grasping` signal + palm-to-block proximity — the `p` variable is no longer used for motion generation
3. IK targets that exceed the reachable workspace envelope are clamped, and convergence failures are logged with frame number
4. Visual overlay of calibrated wrist trajectory on the sim workspace shows consistent axis alignment (no mirroring or rotation)

### Phase 3: Quality Uplift
**Goal:** Upgrade tracking quality with HaMeR (>85% detection) and polish grasp visuals with finger pre-shaping and collision exclusion.
**Requirements:** TRK-04, TRK-05, OUT-03, OUT-04, OUT-05

Two parallel workstreams:

**3a — HaMeR Integration (TRK-04, TRK-05)**
Deploy HaMeR on Modal A10G GPU. Use HaMeR's direct 3D wrist output to bypass depth-map unprojection.

**3b — Grasp Visual Quality (OUT-03, OUT-04, OUT-05)**
Distance-based finger pre-shaping during approach, contype/conaffinity collision exclusion during kinematic hold.

**Success Criteria:**
1. HaMeR achieves >85% hand detection rate on the test recording (vs. MediaPipe's 52%)
2. When HaMeR 3D wrist output is available, depth-map unprojection is skipped entirely
3. Fingers visibly begin closing during descent approach (before contact), proportional to palm-to-block distance
4. No visible finger-block interpenetration during kinematic hold phase
5. STACKED=True still passes with HaMeR tracking and improved grasp visuals

---
*Created: 2026-03-14*
