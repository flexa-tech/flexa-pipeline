# Quick Task 9: Fix calibration scale — anchor wrist to objects

**Created:** 2026-03-14
**Status:** planned

## Tasks

### Task 1: Rewrite calibrate_workspace.py transform logic

**File:** `calibrate_workspace.py`

**Changes:**

1. **Remove arbitrary scale** (lines 116-124) — replace with `scale = 1.0`. R3D data is physical meters from Apple LiDAR; no range mapping needed.

2. **Handle manual objects** — before calling `r3d_to_sim_axes()` on object positions, check each detection's `source` field. If `source == "manual"`, the position is already in sim coords — skip axis swap for those objects. This requires passing the `dets` list metadata through, not just `pos_world`.

3. **Anchor via grasp centroid** — replace the current offset formula:
   - Compute `grasp_centroid_sim_axes = mean(wrist_sim_axes[grasp_frames])` (average wrist position during all grasp frames, after axis swap)
   - Compute `offset = obj_centroid_sim - grasp_centroid_sim_axes` (no scale multiplication since scale=1.0)
   - Apply: `wrist_sim = wrist_sim_axes + offset`

4. **Keep Z correction** — continue forcing object Z to `G1_TABLE_HEIGHT + 0.03` and applying the same Z correction to wrist.

5. **Update saved metadata** — store `scale: 1.0` and `anchor: "grasp_centroid"` in the output JSON for traceability.

### Task 2: Verify end-to-end

Run the verification sequence:
```bash
cd /Users/christian/Documents/ai_dev/flexa-pipeline && source .venv/bin/activate
python calibrate_workspace.py stack2_hamer
python run_pipeline.py --r3d raw_captures/stack2/stack2.r3d \
  --robot g1 --task stack --session stack2_hamer \
  --hamer --trim \
  --objects '[[0.5, 0.01, 0.425], [0.5, -0.01, 0.425]]'
python mujoco_g1_v10.py stack2_hamer
```

**Verify:**
- Calibration output shows `scale: 1.0`
- Wrist X range overlaps G1 workspace [0.10, 0.65] (not crushed to tiny sub-range)
- Wrist Z range is near table height after Z correction
- Manual objects NOT axis-swapped (stay at ~[0.5, 0.0, 0.81])
- Simulation renders (STACKED result noted but not blocking)

### Task 3: Commit

Commit the fix with a concise message referencing the calibration scale change.
