---
phase: 02-core-retargeting
plan: 02
type: execute
wave: 1
depends_on: [01-foundations/01-PLAN, 01-foundations/02-PLAN]
files_modified: [mujoco_g1_v10.py]
autonomous: true
requirements: [RET-01, RET-02, RET-03, RET-04]

must_haves:
  truths:
    - "The `p` progress variable no longer exists in the frame loop"
    - "IK target for ALL frames comes from wrist[i], not choreographed waypoints"
    - "want_grip is derived from grasping[i], not from p-based thresholds"
    - "IK targets are clamped to WORKSPACE_MIN/WORKSPACE_MAX bounds"
    - "IK convergence failures are logged with frame number and error magnitude"
    - "Trajectory workspace stats are printed before the frame loop for axis validation"
    - "Palm tracking positions are recorded and RMS error is computed at the end"
  artifacts:
    - path: "mujoco_g1_v10.py"
      provides: "True wrist retargeting with workspace clamping and convergence logging"
      contains: "np.clip(target, WORKSPACE_MIN, WORKSPACE_MAX)"
    - path: "sim_renders/stack2_g1_v10.mp4"
      provides: "Simulation video driven by real human wrist trajectory"
  key_links:
    - from: "wrist_trajectories/stack2_calibrated.json"
      to: "mujoco_g1_v10.py"
      via: "render_task() reads wrist_sim and grasping arrays"
      pattern: "wrist[i]"
    - from: "mujoco_g1_v10.py"
      to: "sim_renders/stack2_g1_v10.mp4"
      via: "render_task() produces video from retargeted trajectory"
      pattern: "target = wrist[i].copy()"
---

<objective>
Eliminate the choreographed `p` progress variable from `mujoco_g1_v10.py` and drive the robot arm from the real human wrist trajectory for every frame of the simulation. Derive grasp intent from the tracking data's `grasping` signal instead of hardcoded thresholds. Clamp IK targets to the reachable workspace and log convergence failures.

Purpose: The current simulation ignores the human's actual hand motion during the grasp window, replacing it with hardcoded waypoints. This makes the pipeline produce choreographed animation, not true retargeting. Phase 2 fixes this core deficiency.
Output: Modified `mujoco_g1_v10.py` where `target = wrist[i]` for ALL frames, `want_grip = grasping[i]`, with workspace clamping and convergence logging.
</objective>

<execution_context>
@/Users/christian/.claude/get-shit-done/workflows/execute-plan.md
@/Users/christian/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/02-core-retargeting/02-RESEARCH.md

Relevant source file:
@mujoco_g1_v10.py
</context>

<tasks>

<task id="1" name="Add workspace clamp constants and tracking variables">
## What
Add the reachable workspace envelope constants (`WORKSPACE_MIN`, `WORKSPACE_MAX`) and IK failure tracking variables to `mujoco_g1_v10.py`.

## Why
RET-03 requires clamping IK targets to a validated workspace envelope and logging convergence failures. These constants define the safe volume for IK targets. The tracking variables accumulate diagnostic data during the frame loop.

## How

**Step 1: Add workspace constants after line 43 (after `BLEND_FRAMES = 12`).**

Find this block at the top of the file:
```python
FINGER_GAIN_MULTIPLIER = 25.0
BLEND_FRAMES = 12  # smooth blend from rest to hand-tracking
```

Add immediately after it:
```python

# Reachable workspace envelope for IK target clamping (RET-03)
# Derived from G1 right arm reach + table geometry
WORKSPACE_MIN = np.array([0.10, -0.45, 0.80])   # X_min, Y_min, Z_min
WORKSPACE_MAX = np.array([0.65,  0.30, 1.20])   # X_max, Y_max, Z_max
```

**Step 2: Add IK tracking variables after the `last_arm_q = SEED.copy()` line (line 304).**

Find this block inside `render_task()`:
```python
    finger_ctrl = FINGER_OPEN.copy()
    last_arm_q = SEED.copy()
```

Add immediately after it:
```python

    # Tracking variables for RET-03 diagnostics
    ik_failures = []     # list of (frame_idx, error_m) tuples
    n_clamped = 0        # count of frames where target was workspace-clamped
    palm_positions = []  # recorded palm positions for RMS error (RET-01)
```

## Verification
Run: `python -c "import mujoco_g1_v10; print('WORKSPACE_MIN:', mujoco_g1_v10.WORKSPACE_MIN); print('WORKSPACE_MAX:', mujoco_g1_v10.WORKSPACE_MAX)"` -- should print the workspace bounds without error.

## Commit
feat(sim): add workspace envelope constants and IK tracking variables for retargeting
</task>

<task id="2" name="Remove p-choreography and replace with trajectory following">
## What
Delete the `p` progress variable, the choreographed height constants (`z_grasp`, `z_hover`, `z_lift`), and the entire `p`-based motion block (lines 293-295 and 324-390). Replace with direct trajectory following: `target = wrist[i]` with Z floor clamping, workspace clamping, and data-driven `want_grip`.

## Why
RET-01 requires the robot arm to follow the real human wrist trajectory for the entire simulation. RET-02 requires grasp triggering from the `grasping` signal, not `p` thresholds. The `p` variable is the core of the choreographed motion that must be eliminated.

## How

**Step 1: Delete the choreography height constants (lines 293-295).**

Find and delete these three lines:
```python
    z_grasp = G1_TABLE_HEIGHT + BLOCK_HALF - 0.005
    z_hover = z_grasp + 0.12
    z_lift = z_grasp + 0.18
```

**Step 2: Replace the entire `p`-based block inside the frame loop.**

Find this block (lines 323-390 of the original file, starting at `for i in range(n):` and covering everything through the last `want_grip = False`):

```python
    for i in range(n):
        p = 0.0 if (i < ls) else (1.0 if i > le else (i-ls)/win)

        # Place target — use actual block positions from sim
        palm_z_offset = -grasp_offset[2] if grasp_offset is not None else 0.02
        if support_nm is not None:
            place_xy = place_xy0
            block_stack_z = G1_TABLE_HEIGHT + 3 * BLOCK_HALF + 0.002
            place_z = block_stack_z + palm_z_offset
        else:
            place_xy = place_xy0
            place_z = G1_TABLE_HEIGHT + BLOCK_HALF + palm_z_offset

        # Compute IK target and grip intent
        if i < ls:
            # Use real wrist trajectory for natural pre-grasp arm motion
            wrist_pos = wrist[i]
            target = np.array([wrist_pos[0], wrist_pos[1],
                               max(wrist_pos[2], G1_TABLE_HEIGHT + 0.15)])
            want_grip = False
        elif i > le:
            target = np.array([place_xy[0], place_xy[1], z_hover])
            want_grip = False
        else:
            if p < 0.12:
                target = np.array([pick_xy0[0], pick_xy0[1], z_hover])
                want_grip = False
            elif p < 0.25:
                t = smoothstep((p-0.12)/0.13)
                target = np.array([pick_xy0[0], pick_xy0[1], z_hover + (z_grasp - z_hover)*t])
                want_grip = t > 0.5
            elif p < 0.42:
                # Dwell at grasp height — fingers closing, attachment triggers
                target = np.array([pick_xy0[0], pick_xy0[1], z_grasp])
                want_grip = True
            elif p < 0.55:
                t = smoothstep((p-0.42)/0.13)
                target = np.array([pick_xy0[0], pick_xy0[1], z_grasp + (z_lift - z_grasp)*t])
                want_grip = True
            elif p < 0.75:
                t = smoothstep((p-0.55)/0.20)
                mid_xy = (pick_xy0 + place_xy) / 2 + np.array([0.0, -0.08])
                if t < 0.5:
                    t2 = t * 2
                    cx = pick_xy0[0] + (mid_xy[0]-pick_xy0[0])*t2
                    cy = pick_xy0[1] + (mid_xy[1]-pick_xy0[1])*t2
                else:
                    t2 = (t-0.5)*2
                    cx = mid_xy[0] + (place_xy[0]-mid_xy[0])*t2
                    cy = mid_xy[1] + (place_xy[1]-mid_xy[1])*t2
                target = np.array([cx, cy, z_lift])
                want_grip = True
            elif p < 0.88:
                t = smoothstep((p-0.75)/0.13)
                target = np.array([place_xy[0], place_xy[1], z_lift + (place_z-z_lift)*t])
                want_grip = True
            elif p < 0.92:
                # Hold at stack position — block settles onto support
                target = np.array([place_xy[0], place_xy[1], place_z])
                want_grip = True
            elif p < 0.95:
                # Release and retreat slightly up
                target = np.array([place_xy[0], place_xy[1], place_z + 0.04])
                want_grip = False
            else:
                t = smoothstep((p-0.95)/0.05)
                target = np.array([place_xy[0], place_xy[1], place_z + 0.04 + (z_hover-place_z)*t])
                want_grip = False
```

Replace with:
```python
    for i in range(n):
        # === RET-01: Follow real wrist trajectory for ALL frames ===
        target = wrist[i].copy()
        target[2] = max(target[2], G1_TABLE_HEIGHT + 0.02)  # Z floor: 2cm above table

        # === RET-03: Clamp to reachable workspace envelope ===
        target_pre_clamp = target.copy()
        target = np.clip(target, WORKSPACE_MIN, WORKSPACE_MAX)
        if not np.allclose(target, target_pre_clamp, atol=1e-6):
            n_clamped += 1

        # === RET-02: Grasp intent from tracking data ===
        want_grip = bool(grasping[i] > 0)
```

**Important notes:**
- The `for i in range(n):` line stays -- only the body changes.
- Everything after this replacement (IK solve at ~line 392, kinematic attachment at ~line 399, finger control at ~line 455, substeps at ~line 460, rendering at ~line 510) stays EXACTLY as is.
- The `smoothstep()` function (lines 131-133) stays -- it's still used by the kinematic blend at line 486.

## Verification
Run: `grep -n "^\s*p\s*=" mujoco_g1_v10.py` -- should return NO results (the `p` variable is gone).
Run: `grep -n "want_grip.*grasping" mujoco_g1_v10.py` -- should find the new `want_grip = bool(grasping[i] > 0)` line.
Run: `grep -n "target = wrist\[i\]" mujoco_g1_v10.py` -- should find the new trajectory following line.

## Commit
feat(sim): replace p-choreography with real wrist trajectory following (RET-01, RET-02)
</task>

<task id="3" name="Add IK convergence logging and RMS tracking">
## What
After the IK solve in the frame loop, check convergence and record palm positions. After the frame loop ends, compute and print RMS tracking error and IK failure summary.

## Why
RET-03 requires convergence failure logging with frame numbers. RET-01 success criterion requires <5cm RMS tracking error. We need to measure both.

## How

**Step 1: Add palm position recording and IK convergence check after the IK solve.**

Find this block inside the frame loop (the IK solve and the start of attachment logic):
```python
        # IK solve
        for qa, v in zip(arm_qa, last_arm_q):
            data.qpos[qa] = v
        mujoco.mj_forward(model, data)
        arm_q = ik_solve(model, data, target, arm_qa, arm_da)
        last_arm_q = arm_q.copy()

        # --- Kinematic block attachment logic (Franka v9 pattern) ---
        pc = palm_center(model, data)
```

Replace with:
```python
        # IK solve
        for qa, v in zip(arm_qa, last_arm_q):
            data.qpos[qa] = v
        mujoco.mj_forward(model, data)
        arm_q = ik_solve(model, data, target, arm_qa, arm_da)
        last_arm_q = arm_q.copy()

        # --- RET-01/RET-03: Track palm position and check IK convergence ---
        pc = palm_center(model, data)
        palm_positions.append(pc.copy())
        ik_err = np.linalg.norm(target - pc)
        if ik_err > 0.02:  # 2cm convergence threshold
            ik_failures.append((i, float(ik_err)))
```

Note: The `pc = palm_center(model, data)` line already existed for the attachment logic. We're adding the recording and convergence check between the IK solve and the attachment logic. The `pc` variable is reused by the attachment logic below.

**Step 2: Add RMS tracking error and convergence summary after the frame loop.**

Find this block after the frame loop (the final position printout):
```python
    for nm, qa in zip(obj_names, obj_qadr):
        pos = data.qpos[qa:qa+3]
        print(f"FINAL {nm}: xyz=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
```

Add BEFORE it (after the frame loop ends, before final stats):
```python
    # === RET-01: Compute RMS tracking error ===
    palm_arr = np.array(palm_positions)
    target_arr = np.clip(wrist.copy(), WORKSPACE_MIN, WORKSPACE_MAX)
    target_arr[:, 2] = np.maximum(target_arr[:, 2], G1_TABLE_HEIGHT + 0.02)
    tracking_errors = np.linalg.norm(palm_arr - target_arr, axis=1)
    rms_err = float(np.sqrt(np.mean(tracking_errors**2)))
    mean_err = float(np.mean(tracking_errors))
    max_err = float(np.max(tracking_errors))
    print(f"\n  RET-01 Tracking: RMS={rms_err:.4f}m mean={mean_err:.4f}m max={max_err:.4f}m")
    if rms_err < 0.05:
        print(f"  RET-01 PASS: RMS error {rms_err:.4f}m < 0.05m threshold")
    else:
        print(f"  RET-01 WARNING: RMS error {rms_err:.4f}m exceeds 0.05m threshold")

    # === RET-03: Log clamping and convergence stats ===
    if n_clamped > 0:
        print(f"  RET-03 Clamping: {n_clamped}/{n} frames clamped to workspace envelope")
    else:
        print(f"  RET-03 Clamping: 0/{n} frames clamped (all targets within workspace)")
    if ik_failures:
        print(f"  RET-03 IK failures: {len(ik_failures)}/{n} frames with >2cm error")
        for frame, err in ik_failures[:5]:
            print(f"    Frame {frame:03d}: err={err:.4f}m")
        if len(ik_failures) > 5:
            print(f"    ... and {len(ik_failures)-5} more")
    else:
        print(f"  RET-03 IK failures: 0/{n} frames (all converged within 2cm)")

```

## Verification
Run: `python mujoco_g1_v10.py stack2 2>&1 | grep "RET-0"` -- should print RET-01 tracking stats and RET-03 clamping/convergence stats.

## Commit
feat(sim): add IK convergence logging and RMS tracking error measurement (RET-01, RET-03)
</task>

<task id="4" name="Add trajectory workspace validation and clean up logging">
## What
Add trajectory workspace statistics before the frame loop for RET-04 visual axis validation. Update the periodic frame log to remove the now-deleted `p` variable. Clean up any remaining references to removed variables.

## Why
RET-04 requires visual overlay validation of the calibrated wrist trajectory on the sim workspace. Printing trajectory bounds and verifying block position overlap provides axis alignment validation. The `p` variable references in logging must be removed since `p` no longer exists.

## How

**Step 1: Add trajectory workspace stats before the frame loop.**

Find this line before the frame loop:
```python
    print(f"\nTrajectory: {n} frames, grip [{ls},{le}], pick={pick_nm}, support={support_nm}")
```

Replace with:
```python
    print(f"\nTrajectory: {n} frames, grip [{ls},{le}], pick={pick_nm}, support={support_nm}")

    # === RET-04: Trajectory workspace validation ===
    print(f"  Wrist X: [{wrist[:,0].min():.3f}, {wrist[:,0].max():.3f}]")
    print(f"  Wrist Y: [{wrist[:,1].min():.3f}, {wrist[:,1].max():.3f}]")
    print(f"  Wrist Z: [{wrist[:,2].min():.3f}, {wrist[:,2].max():.3f}]")
    print(f"  Workspace clamp: X[{WORKSPACE_MIN[0]:.2f},{WORKSPACE_MAX[0]:.2f}] "
          f"Y[{WORKSPACE_MIN[1]:.2f},{WORKSPACE_MAX[1]:.2f}] "
          f"Z[{WORKSPACE_MIN[2]:.2f},{WORKSPACE_MAX[2]:.2f}]")
    pick_in_x = wrist[:,0].min() <= desired_pick_xy[0] <= wrist[:,0].max()
    pick_in_y = wrist[:,1].min() <= desired_pick_xy[1] <= wrist[:,1].max()
    print(f"  Pick block ({desired_pick_xy.round(3)}) within wrist XY range: "
          f"X={'yes' if pick_in_x else 'NO'} Y={'yes' if pick_in_y else 'NO'}")
```

**Step 2: Update the periodic frame log to remove `p`.**

Find this line inside the frame loop (in the `if i % 10 == 0:` block):
```python
            print(f"F{i:03d} p={p:.2f} grip={want_grip} attached={attached} palm={pc.round(3)} "
```

Replace with:
```python
            print(f"F{i:03d} grip={want_grip} attached={attached} palm={pc.round(3)} "
```

**Step 3: Verify no remaining references to `p`, `z_hover`, `z_lift`, `z_grasp`, `place_xy`, `place_z`, or `palm_z_offset`.**

After making all changes, verify these variables are not referenced anywhere in the frame loop:
- `p` -- deleted (was line 324)
- `z_hover` -- deleted (was line 294)
- `z_lift` -- deleted (was line 295)
- `z_grasp` -- deleted (was line 293)
- `place_xy` -- deleted (was lines 329, 333)
- `place_z` -- deleted (was lines 331, 334)
- `palm_z_offset` -- deleted (was line 327)
- `pick_xy0` -- still referenced in block positioning (line 283) and log (line 273), but NOT in IK target computation. This is correct.
- `place_xy0` -- still referenced in block positioning (line 288). This is correct.

## Verification
Run: `python mujoco_g1_v10.py stack2 2>&1 | grep "RET-04"` -- should NOT appear (it's just a print label in the code comment, not output). Instead verify:
Run: `python mujoco_g1_v10.py stack2 2>&1 | grep "Wrist X"` -- should print wrist coordinate ranges.
Run: `python mujoco_g1_v10.py stack2 2>&1 | grep "Pick block"` -- should print whether pick block is within wrist range.
Run: `grep "p={p" mujoco_g1_v10.py` -- should return NO results.

## Commit
feat(sim): add trajectory workspace validation and clean up p-variable references (RET-04)
</task>

<task id="5" name="Run full validation and measure tracking accuracy">
## What
Run the simulation on stack2 with the retargeted code and verify all four RET requirements.

## Why
All code changes are complete. This task validates that the retargeted simulation meets the success criteria: <5cm RMS error (RET-01), data-driven grasp triggering (RET-02), workspace clamping with logging (RET-03), and axis alignment (RET-04).

## How

**Step 1: Ensure calibrated data is fresh (re-calibrate + re-trim if needed).**

Check if the calibrated JSON exists and has trim_info:
```bash
cd /Users/christian/Documents/ai_dev/flexa-pipeline && python -c "
import json
with open('wrist_trajectories/stack2_calibrated.json') as f:
    calib = json.load(f)
print(f'Frames: {len(calib[\"wrist_sim\"])}')
print(f'Trim info: {calib.get(\"trim_info\", \"NONE\")}')
print(f'Grasping frames: {sum(1 for g in calib[\"grasping\"] if g)}')
"
```

If `Frames` is 951 (untrimmed), re-run calibration and trimming:
```bash
python calibrate_workspace.py stack2 && python trim_trajectory.py stack2
```

If `Frames` is ~229 (already trimmed), proceed to Step 2.

**Step 2: Run the simulation.**
```bash
cd /Users/christian/Documents/ai_dev/flexa-pipeline && python mujoco_g1_v10.py stack2
```

**Step 3: Validate output against success criteria.**

Check the simulation output for:

1. **RET-01:** Look for `RET-01 Tracking: RMS=X.XXXXm`. Must be < 0.05m.
   - If RMS > 0.05m, check how many IK failures there are. High failure count suggests workspace bounds are too tight or trajectory is out of reach.

2. **RET-02:** Verify `want_grip` changes are visible in the frame log. Look for frames where `grip=True` transitions to `grip=False` and vice versa. These should correspond to the grasping signal in the calibrated data, NOT to fixed progress thresholds.
   - Run: `grep "p=" mujoco_g1_v10.py` -- should return NO lines inside the frame loop (the `p` variable is gone).

3. **RET-03:** Look for clamping and IK failure stats in output:
   - `RET-03 Clamping: X/N frames clamped`
   - `RET-03 IK failures: X/N frames`

4. **RET-04:** Look for trajectory workspace stats:
   - `Wrist X: [min, max]`
   - `Pick block (...) within wrist XY range: X=yes Y=yes`
   - If either is `NO`, the axis mapping may be wrong. Check `calibrate_workspace.py`'s `r3d_to_sim_axes()`.

**Step 4: Check STACKED status.**

The output will print `STACKED=True` or `STACKED=False`. With true retargeting, STACKED may be False because the robot now follows the human's actual trajectory instead of the choreographed pick-and-place. This is EXPECTED and ACCEPTABLE for Phase 2. Document the result.

If STACKED=True, that's a bonus -- it means the calibration aligned the trajectory well enough for the human's actual motion to produce a successful stack.

**Step 5: If RMS error exceeds 5cm, diagnose.**

Common causes:
1. **Many frames clamped:** Trajectory extends beyond workspace bounds. Consider relaxing WORKSPACE_MIN/WORKSPACE_MAX slightly.
2. **Many IK failures:** Targets are reachable but IK solver doesn't converge. Check if the convergence threshold (0.004m in `ik_solve`) is too tight, or if the max iterations (150) is too few.
3. **Z floor too restrictive:** If wrist Z in calibrated data is consistently below 0.80, the Z floor clamp is pulling targets up, causing systematic error. Lower Z_min in WORKSPACE_MIN.

If adjustments are needed, modify only the constants (WORKSPACE_MIN, WORKSPACE_MAX) and re-run. Do NOT reintroduce choreographed motion.

## Verification
- Simulation completes without errors
- `RET-01 PASS` or `RET-01 WARNING` appears in output (with RMS error value)
- `RET-03 Clamping` and `RET-03 IK failures` stats appear in output
- `Wrist X/Y/Z` ranges appear in output
- `Pick block ... within wrist XY range` appears in output
- The `p` variable does not appear in any grep of mujoco_g1_v10.py (except possibly comments)

## Commit
test(sim): validate phase 2 retargeting on stack2 data
</task>

</tasks>

<verification>
Before declaring plan complete:
- [ ] `grep -n "^\s*p\s*=" mujoco_g1_v10.py` returns no results (p variable eliminated)
- [ ] `grep -n "z_hover\|z_lift\|z_grasp" mujoco_g1_v10.py` returns no results (choreography heights removed)
- [ ] `grep -n "target = wrist\[i\]" mujoco_g1_v10.py` finds the trajectory following line
- [ ] `grep -n "want_grip.*grasping\[i\]" mujoco_g1_v10.py` finds the data-driven grasp line
- [ ] `grep -n "WORKSPACE_MIN\|WORKSPACE_MAX" mujoco_g1_v10.py` finds the clamp constants
- [ ] `grep -n "ik_failures" mujoco_g1_v10.py` finds convergence logging
- [ ] `grep -n "palm_positions" mujoco_g1_v10.py` finds RMS tracking recording
- [ ] Simulation output shows `RET-01 Tracking: RMS=` with a value
- [ ] Simulation output shows `RET-03 Clamping:` stats
- [ ] Simulation output shows `Wrist X:` trajectory ranges
- [ ] Simulation completes without Python errors
</verification>

<success_criteria>
- The `p` progress variable is completely removed from the frame loop
- IK target for ALL frames comes from `wrist[i]` (the real human trajectory)
- `want_grip` derives from `grasping[i]` (tracking data), not from `p` thresholds
- IK targets are clamped to WORKSPACE_MIN/WORKSPACE_MAX before solving
- Convergence failures (>2cm error after IK solve) are logged with frame numbers
- RMS tracking error is computed and printed (<5cm target)
- Trajectory workspace ranges are printed before the frame loop for axis validation
- Pick block position overlap with wrist trajectory range is verified
- Simulation runs to completion on stack2 trimmed data without errors
- Note: STACKED=True is not a requirement for Phase 2 -- true retargeting accuracy is the primary goal
</success_criteria>

<output>
After completion, create `.planning/phases/02-core-retargeting/02-SUMMARY.md`
</output>
