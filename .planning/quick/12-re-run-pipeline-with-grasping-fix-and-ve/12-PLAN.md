# QT-012: Re-run pipeline with grasping fix and verify sim

## Task

Run the full pipeline and simulation after QT-011's grasping signal cleanup (clean_grasping_signal + 2cm HaMeR threshold). Verify that grasping % dropped from ~66% and check STACKED result.

## Steps

### 1. Run pipeline and analyze grasping output
```bash
cd /Users/christian/Documents/ai_dev/flexa-pipeline && source .venv/bin/activate
python run_pipeline.py --r3d raw_captures/stack2/stack2.r3d \
  --robot g1 --task stack --session stack2_hamer \
  --hamer --trim \
  --objects '[[0.5, 0.01, 0.425], [0.5, -0.01, 0.425]]'
```
- HaMeR stage will use cached Modal results (skip to calibration + trim + sim stages)
- Check `clean_grasping_signal()` stats in output — grasping % should drop significantly from raw ~66%
- Note trim window, frame count, and any errors

### 2. Run simulation and report findings
```bash
python mujoco_g1_v10.py stack2_hamer
```
- Check STACKED=True/False
- Observe finger behavior: do fingers open during approach, then close on grasp?
- Report what changed vs previous runs

## Success Criteria
- Pipeline completes all stages without error
- Grasping signal cleaned (% significantly lower than 66%)
- STACKED result documented with analysis of what changed
