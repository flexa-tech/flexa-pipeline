# Quick Task 9: Fix calibration scale — anchor wrist to objects

**Completed:** 2026-03-14
**Commit:** 7fbe221
**Branch:** fix/e2e-pipeline

## Changes

Rewrote `calibrate_workspace.py` transform logic:

1. **Scale 1.0** — removed `wrist_range / 0.35` computation (was producing scale=0.222, crushing trajectory to 22% of real size). R3D depth data is physical meters from Apple LiDAR; no rescaling needed.

2. **Manual object handling** — loads full detection dicts and checks `source` field. Manual objects (`source == "manual"`) skip `r3d_to_sim_axes()` since they're already in sim coordinates. Detected objects (GroundingDINO) still get axis-swapped.

3. **Grasp centroid anchor** — computes average wrist position during all grasp frames (in sim axes), uses as anchor point. Offset = `obj_centroid_sim - grasp_centroid`. Previously offset was `obj_centroid_sim - obj_centroid_sim_axes * scale` which didn't tie wrist to objects.

4. **Z correction preserved** — objects forced to `G1_TABLE_HEIGHT + 0.03 = 0.81m`, same correction applied to wrist.

5. **Metadata** — output JSON now stores `scale: 1.0`, `anchor: "grasp_centroid"`, and `grasp_centroid` coordinates.

## Calibration Output Comparison

| Metric | Before (QT-8) | After (QT-9) |
|--------|---------------|---------------|
| Scale | 0.222 | 1.000 |
| Anchor | obj centroid offset | grasp centroid |
| Wrist X range | compressed ~0.35m | [-0.613, 0.964] |
| Wrist Y range | compressed | [-0.650, 0.347] |
| Wrist Z range | compressed | [0.267, 1.125] |
| Objects Z | 0.425 | 0.810 |

## Simulation Result

- **STACKED = False**
- Block separation: 0.368m (blocks placed far apart due to large wrist trajectory range)
- RMS tracking error: 0.1065m (mean=0.074m, max=0.215m)
- 93/300 frames clamped to workspace bounds
- 155/300 frames with IK error > 2cm
- Hand never reaches pick block — wrist trajectory covers large range but grasping signal is all-True for the trimmed window (no approach phase distinction)

## Root Cause Analysis

The calibration fix is correct: scale=1.0 preserves physical distances and grasp centroid anchoring ties the wrist to objects. However STACKED=False persists because:

1. **Block placement uses wrist trajectory extremes** — pick block placed at the wrist XY during earliest grasp frame, support block at latest. With scale=1.0 the wrist covers a much larger range, placing blocks 0.368m apart.
2. **Entire trimmed window is grip=True** — the trim window [392,692) still has grasping=True for nearly all frames, so there's no clear approach vs manipulation phase for the sim to distinguish.
3. **IK/workspace clamping** — human arm range exceeds G1 workspace; 31% of frames get clamped.

## Next Steps

The calibration transform is now physically grounded. Remaining bottlenecks are:
- Block placement logic in sim (should use manual object positions directly, not derive from wrist trajectory)
- Grasping signal quality (HaMeR aspect-ratio heuristic produces mostly-True signal)
- IK convergence with large wrist ranges
