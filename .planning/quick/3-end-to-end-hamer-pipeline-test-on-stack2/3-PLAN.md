# Quick Task 3: End-to-End HaMeR Pipeline Test on Stack2

## Objective
Run the full pipeline with `--hamer` on stack2.r3d, fix any bugs encountered, then render simulation video.

## Tasks

### Task 1: Run HaMeR Pipeline & Fix Bugs
**Command:**
```bash
cd /Users/christian/Documents/ai_dev/flexa-pipeline && \
source .venv/bin/activate && \
python run_pipeline.py \
  --r3d raw_captures/stack2/stack2.r3d \
  --robot g1 --task stack \
  --session stack2_hamer \
  --hamer --trim \
  --objects '[[0.5, 0.01, 0.425], [0.5, -0.01, 0.425]]'
```

**Expected issues (first real test):**
- Modal function `flexa-hamer` may not be deployed yet -> deploy with `modal deploy processing/hamer_modal.py`
- `modal.Function.lookup("flexa-hamer", "run_hamer_inference")` may fail if function not found -> fix auth/naming
- HaMeR `wrist_3d_camera` values may be in MANO-relative coords (not true camera frame) -> `cam_to_world` transform could produce wrong world positions
- Frame count mismatch: R3D 60fps vs pipeline 30fps -> `fps_ratio` mapping in reconstruct_wrist_3d.py
- Interpolation assertion may fail if HaMeR detection rate leaves edge NaN gaps

**Fix pattern:** Run pipeline, read error, fix code, re-run. Iterate until stage 6 (trim) completes.

### Task 2: Run Simulation & Verify Output
**Command:**
```bash
cd /Users/christian/Documents/ai_dev/flexa-pipeline && \
source .venv/bin/activate && \
python mujoco_g1_v10.py stack2_hamer
```

**Success criteria:**
- Video renders without crash
- Robot tracks a plausible wrist trajectory (visual check)
- Compare detection rate vs MediaPipe baseline (52% from STATE.md)

### Task 3: Commit Results
- Commit any bug fixes made during Task 1
- Update STATE.md with HaMeR test results (detection rate, model mode, any remaining issues)

## Key Files
- `run_pipeline.py` — orchestrator, `_run_hand_tracking` + `_hamer_to_trajectory`
- `egocrowd/hand_pose.py` — Modal remote call to HaMeR
- `processing/hamer_modal.py` — GPU function (GroundingDINO + HaMeR)
- `reconstruct_wrist_3d.py` — `cam_to_world` for HaMeR 3D path
- `calibrate_workspace.py` — R3D-to-sim transform
- `mujoco_g1_v10.py` — simulation renderer

## Exit Criteria
- Pipeline completes all 7 stages with `--hamer` flag
- Simulation video exists at `sim_renders/stack2_hamer_*.mp4`
- All bug fixes committed
