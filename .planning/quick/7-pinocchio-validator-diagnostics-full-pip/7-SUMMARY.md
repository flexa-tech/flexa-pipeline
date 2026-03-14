# QT-007 Summary: Pinocchio Validator Diagnostics + Full Pipeline Re-test

**Date:** 2026-03-14
**Branch:** fix/e2e-pipeline
**Result:** STACKED=False

---

## 1. Pinocchio Validator Results

Ran on `wrist_trajectories/stack2_hamer_calibrated.json` (pre-pipeline, 951 frames):

| Metric | HaMeR (calibrated) | MediaPipe baseline |
|--------|--------------------|--------------------|
| Mean IK error | 0.1083m | 0.115m |
| Median IK error | 0.1154m | — |
| P95 IK error | 0.1774m | — |
| Max IK error | 0.2064m | — |
| Converged (<4mm) | 4.3% (41/951) | — |
| Usable (<5cm) | 16.3% (155/951) | — |
| >10cm error | 68.1% (648/951) | — |

**Assessment:** HaMeR calibrated trajectory is only marginally better than MediaPipe (0.1083m vs 0.115m). The overwhelming majority of frames (68%) have >10cm IK error, meaning the wrist targets are largely unreachable by the G1 arm. Joint limit violations: 18 frames.

## 2. Full Pipeline Results

All 7 stages completed successfully in 401.5s:

| Stage | Status | Notes |
|-------|--------|-------|
| 1. Ingest R3D | OK | 951 frames, 960x720 @ 60fps |
| 2. Hand tracking (HaMeR) | OK | 941/951 frames detected (98.9%) |
| 3. Object detection | OK | 2 manual objects |
| 4. 3D wrist reconstruction | OK | 941/951 valid, depth unprojection |
| 5. Workspace calibration | OK | Scale=0.222, wrist X=[0.196, 0.546] |
| 6. Trim trajectory | OK | 951 → 269 frames (26.9s), window [568, 837) |
| 7. Simulation | OK | Video rendered, 269 frames |

Wrist range after calibration: X=[0.196, 0.546], Y=[0.026, 0.247], Z=[0.687, 0.877].

## 3. Simulation Results

**STACKED=False**

Final block positions:
- block_a (pick): (0.338, -0.062, 0.810) — at table height, unmoved
- block_b (support): (0.493, 0.178, 0.810) — at table height, unmoved

Stack check: top_z=0.810, bot_z=0.810, gap=0.000 (expected 0.060). XY dist=0.286m (threshold 0.060m).

### Simulation diagnostics:
- **RMS tracking error:** 0.0670m (exceeds 0.05m threshold)
- **IK failures:** 186/269 frames with >2cm error (69%)
- **Wrist clamped to Z >= 0.80:** wrist Z range is [0.687, 0.877], but workspace clamp floors Z at 0.80
- **All 269 frames have grip=True:** the trim window [568, 837) falls entirely within the grasping region (632 grasp frames), so the robot is "gripping" from frame 0 but never performs a reach-to-grasp approach

## 4. Root Cause Analysis

### Why STACKED=False:

1. **Wrist never reaches the block.** Minimum palm-to-block_a distance was 0.034m (frame 0), but quickly diverges to 0.13-0.22m. The palm oscillates in a region above and to the side of the pick block rather than approaching it.

2. **Trim window captures only grip-phase.** The trimmed window [568, 837) is entirely within the grasping region. There is no approach phase — the robot starts gripping immediately. Without an approach-then-grasp sequence, the physics-based grasping system never has a chance to close fingers around the block.

3. **Systematic Z offset persists.** Raw wrist Z values (0.687-0.877m) span a wide range, but many frames fall below the table (0.810m). The workspace Z clamp at 0.80m pushes these up, but the IK solver still struggles to track the lateral (XY) component, resulting in 69% of frames exceeding 2cm IK error.

4. **Calibration scale too aggressive.** The 0.222 scale factor compresses the full camera-space motion into a narrow sim-space band. The wrist path becomes a tight oscillation around a fixed point rather than a deliberate reach-to-target motion.

### Comparison to MediaPipe path:
The MediaPipe trajectory had similar IK error (0.115m) but achieved STACKED=True in earlier tests because the trim window captured an approach+grasp+lift sequence. The HaMeR trajectory's 98.9% detection rate (vs 52% MediaPipe) gives more frames, but the higher frame count shifts the grasping density window to a different region that lacks the approach phase.

### Next steps (not part of this task):
- Fix trim logic to ensure approach phase is included (pre-grasp frames)
- Investigate calibration scale — 0.222 may be compressing motion too aggressively
- Consider separate reach/grasp phase detection in trim module
