# Requirements: Flexa Pipeline

**Defined:** 2026-03-14
**Core Value:** The robot must faithfully reproduce the human's actual hand motion — true retargeting, not choreographed animation.

## v0.3 Requirements

Requirements for True Retargeting & Quality milestone. Each maps to roadmap phases.

### Retargeting

- [ ] **RET-01**: Robot arm follows real human wrist trajectory for entire simulation, not just pre-grasp
- [ ] **RET-02**: Grasp triggering driven by tracking data `grasping` signal + palm-to-block proximity, not choreographed `p` variable
- [ ] **RET-03**: IK targets clamped to validated reachable workspace envelope with convergence failure logging
- [ ] **RET-04**: Coordinate frame mapping validated visually (calibrated trajectory overlaid on sim workspace)

### Tracking Quality

- [ ] **TRK-01**: Trajectory gaps interpolated using gap-length-aware strategy (cubic spline <15 frames, PCHIP 15-30, linear >30)
- [ ] **TRK-02**: Spatial trajectory smoothed with Savitzky-Golay filter (zero-phase, preserves grasp dwell)
- [ ] **TRK-03**: Grasping signal preserved as binary (never smoothed)
- [ ] **TRK-04**: HaMeR hand pose estimation deployed on Modal GPU with >85% detection rate
- [ ] **TRK-05**: HaMeR 3D wrist output used directly (skip depth-map unprojection when available)

### Output Quality

- [ ] **OUT-01**: Video auto-trimmed to action window (first grasp - 30 frames to last grasp + 50 frames)
- [ ] **OUT-02**: Output video is 15-30 seconds (not 95 seconds)
- [ ] **OUT-03**: Fingers pre-shape during descent approach based on palm-to-block distance
- [ ] **OUT-04**: Finger-block collision excluded during kinematic hold via contype/conaffinity
- [ ] **OUT-05**: STACKED=True still passes after all changes

## Future Requirements

### Advanced Retargeting

- **RET-F01**: Multi-hand manipulation support
- **RET-F02**: Real-time retargeting (streaming, not batch)
- **RET-F03**: Support for non-G1 robot models

### Advanced Tracking

- **TRK-F01**: Finger-level retargeting (individual finger trajectories)
- **TRK-F02**: Object pose tracking (6-DOF block state)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Dynamic grasping physics | Kinematic attachment sufficient for research stage |
| Real-time inference | Batch processing acceptable for current use case |
| Multi-robot support | G1 is sole target platform |
| Blending MediaPipe + HaMeR per-frame | Research pitfall P5 — coordinate frame mismatch causes artifacts |
| pytorch3d dependency | Not needed by HaMeR core inference |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| TRK-01 | — | Pending |
| TRK-02 | — | Pending |
| TRK-03 | — | Pending |
| RET-01 | — | Pending |
| RET-02 | — | Pending |
| RET-03 | — | Pending |
| RET-04 | — | Pending |
| OUT-01 | — | Pending |
| OUT-02 | — | Pending |
| TRK-04 | — | Pending |
| TRK-05 | — | Pending |
| OUT-03 | — | Pending |
| OUT-04 | — | Pending |
| OUT-05 | — | Pending |

**Coverage:**
- v0.3 requirements: 14 total
- Mapped to phases: 0
- Unmapped: 14 ⚠️

---
*Requirements defined: 2026-03-14*
*Last updated: 2026-03-14 after initial definition*
