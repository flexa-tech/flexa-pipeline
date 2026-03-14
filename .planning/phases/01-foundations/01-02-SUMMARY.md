---
phase: 01-foundations
plan: 02
subsystem: pipeline
tags: [trimming, action-detection, velocity, grasping, numpy]

# Dependency graph
requires:
  - phase: 01-foundations/01
    provides: "Calibrated wrist trajectory JSON with smoothed wrist_sim and binary grasping"
provides:
  - "trim_trajectory.py module with action window auto-detection"
  - "Pipeline --trim flag for optional trimming between calibration and simulation"
  - "trim_info traceability metadata in calibrated JSON"
affects: [02-core-retargeting, 03-quality-uplift]

# Tech tracking
tech-stack:
  added: []
  patterns: [sliding-window-density, in-place-json-modification, fallback-chain]

key-files:
  created: [trim_trajectory.py]
  modified: [run_pipeline.py]

key-decisions:
  - "Trim modifies calibrated JSON in-place (no new file) to keep downstream stages unchanged"
  - "Added sliding-window cluster focus when action_frames span exceeds target_max (300 frames)"
  - "Late grasp cluster (frames 547-776) is the correct action window for stack2"
  - "Trim stage is opt-in (--trim flag) for backward compatibility"

patterns-established:
  - "Action detection: combined velocity + grasping signal with 3-level fallback chain"
  - "Duration enforcement: sliding window finds densest action cluster when span too wide"
  - "Traceability: trim_info metadata in JSON records original_frames, start_frame, end_frame"

requirements-completed: [OUT-01, OUT-02]

# Metrics
duration: 15min
completed: 2026-03-14
---

# Phase 1, Plan 02: Video Trimming Summary

**Action window auto-detection trims stack2 from 95s to 22.9s using velocity + grasping density, with STACKED=True regression passing**

## Performance

- **Duration:** 15 min
- **Tasks:** 3
- **Files created:** 1
- **Files modified:** 1

## Accomplishments
- Auto-detect action window from grasping signal + wrist velocity with 3-level fallback chain
- Trim 95-second recording to 22.9-second action window (within 15-30s target)
- STACKED=True regression passes on trimmed data
- Pipeline CLI supports --trim, --trim-start, --trim-end flags

## Task Commits

Each task was committed atomically:

1. **Task 1: Create trim_trajectory.py** - `8b4bdaa` (feat)
2. **Task 2: Integrate trim stage into run_pipeline.py** - `764b014` (feat)
3. **Task 3: Validate and fix duration enforcement** - `036655e` (fix)

## Files Created/Modified
- `trim_trajectory.py` - Action window detection + calibrated data trimming module
- `run_pipeline.py` - Added trim stage 6 between calibration and simulation, total stages 6->7

## Decisions Made
- **In-place JSON modification:** Trim overwrites the calibrated JSON rather than creating a new file, keeping all downstream stages (simulation) unchanged.
- **Sliding-window cluster focus:** The plan's original tightening logic only reduced margins, which was insufficient when action_frames spanned 534 frames. Added a sliding-window approach that finds the densest action cluster to enforce the 300-frame max.
- **Opt-in trimming:** --trim flag defaults to False, so existing workflows are unaffected.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Duration enforcement logic insufficient for sparse action data**
- **Found during:** Task 3 (stack2 validation)
- **Issue:** Plan's tightening logic only reduced margins (~50 frames), but action_frames spanned 534 frames (88-726). Output was 578 frames (57.8s), far exceeding the 15-30s target.
- **Fix:** Added sliding-window density search in detect_action_window() that finds the best target_max-sized window containing the most action frames, then focuses on that cluster.
- **Files modified:** trim_trajectory.py
- **Verification:** stack2 now trims to 229 frames (22.9s), STACKED=True passes
- **Committed in:** 036655e (Task 3 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Essential fix for correctness. The plan anticipated this possibility in Task 3, Step 5 ("adjust margin_before or margin_after in detect_action_window()"). No scope creep.

## Issues Encountered
None beyond the auto-fixed deviation above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 1 (Foundations) complete: both trajectory smoothing (Plan 01) and video trimming (Plan 02) are done
- Trimmed 22.9s output significantly reduces iteration time for Phase 2 (Core Retargeting)
- trim_info metadata enables traceability for debugging retargeting issues

---
*Phase: 01-foundations*
*Completed: 2026-03-14*
