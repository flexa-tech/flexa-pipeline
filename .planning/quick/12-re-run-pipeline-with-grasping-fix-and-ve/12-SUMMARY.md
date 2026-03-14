# QT-012 Summary: Re-run pipeline with grasping fix and verify sim

## Result: STACKED=False

## GRASP CLEAN Stats

- **Before**: 630/951 (66%) frames marked as grasping
- **After clean_grasping_signal()**: 558/951 (59%)
- **Reduction**: Only 7 percentage points (72 frames removed)
- **Onset frame**: 86 (raw, pre-trim)
- **Post-debounce in sim**: 226 raw -> 270 debounced frames (out of 300 total)

The temporal filter is not aggressive enough. The grasping signal is still True for 90% of the trimmed trajectory.

## Trim Window

- **Window**: [407, 707) — 300 frames, 30.0s
- **Original**: 951 frames (95.1s)

## Simulation Analysis

- **STACKED=False**
- **Pick block**: block_a at (0.5, 0.01, 0.81)
- **Final positions**: block_a fell off table (z=0.030), block_b stayed (z=0.810)

### Failure sequence

1. F000-F020: grip=False, palm moving away from blocks (approach phase exists but is only 3 frames)
2. F030: grip=True onset — palm at (0.098, 0.280, 0.813), **0.334m from block_a** (never near the block)
3. F040-F120: grip=True, hand sweeps around with fingers closed, occasionally passes near blocks
4. F120: palm at (0.431, -0.103, 0.812), closest approach to block_a at 0.092m — still not close enough for a grasp
5. F130: block_a z=0.818 (slightly bumped), distance jumps to 0.507m
6. F140: block_a z=0.030 (knocked off table by closed-fist sweep)
7. F140-F290: block_a on the ground, hand continues moving with grip=True

### Root causes

1. **Grasping signal still overwhelmingly True**: 90% of trimmed frames have grip=True. The clean_grasping_signal filter only removed 72 frames globally (66%->59%). After trimming to [407,707) and debouncing, only frames 0-29 are grip=False.
2. **No close approach before grasp**: The hand never gets within grasping distance of block_a before grip=True fires. At grip onset (F030), palm-to-block distance is 0.334m.
3. **IK tracking poor**: RMS=0.099m (2x the 0.05m threshold), 146/300 frames with >2cm IK error, 105/300 frames workspace-clamped.

## Grasping Transition Visibility

**No** — the trimmed data does not show a clean False->True transition near the block. The grasping signal fires True at F030 when the hand is far from any block (0.334m). This is a HaMeR detection artifact — the aspect-ratio grasping heuristic is triggering on hand poses that aren't actually grasping.

## Next Steps

1. **Fix the grasping heuristic** — The aspect-ratio based grasping detection from HaMeR bounding boxes is unreliable. Consider:
   - Using HaMeR's actual hand pose (finger joint angles) to determine grasp state instead of bbox aspect ratio
   - Proximity-gated grasping: only allow grip=True when palm is within N cm of an object
   - Much more aggressive temporal filtering (e.g., require 10+ consecutive True frames to count)
2. **Fix IK/calibration** — RMS=0.099m is too high. The wrist trajectory doesn't bring the hand close enough to the block. The calibration anchor (grasp centroid) may be averaging over too many spurious grasping frames, pulling the anchor away from the actual grasp location.
3. **Consider grasping from sim proximity** — Rather than trusting the vision-derived grasping signal, derive grasp events from palm-to-block distance in the sim. When the IK-driven palm is within 3cm of a block AND the vision signal agrees, close fingers.
