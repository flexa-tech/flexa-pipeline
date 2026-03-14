# Phase 2: Core Retargeting -- Research

*Researched: 2026-03-14*
*Scope: RET-01, RET-02, RET-03, RET-04*

---

## 1. Current Code Analysis

### The `p` Progress Variable -- What It Does and Why It Must Go

**File:** `mujoco_g1_v10.py`, lines 323-390

The `p` variable is a normalized [0.0, 1.0] progress scalar that maps the grasp window `[ls, le]` into choreographed motion phases:

```
Line 324: p = 0.0 if (i < ls) else (1.0 if i > le else (i-ls)/win)
```

Where `ls` and `le` are the first/last frame indices of "late" grasping signals (frames >60% of trajectory length), and `win = le - ls`.

**Phase breakdown of the `p` choreography (lines 337-390):**

| `p` range | Phase | What happens | Lines |
|-----------|-------|-------------|-------|
| `i < ls` | Pre-grasp | Uses real `wrist[i]` trajectory, Z clamped to TABLE+0.15 | 337-341 |
| 0.00-0.12 | Hover | IK target at `pick_xy0` at `z_hover` | 347-349 |
| 0.12-0.25 | Descend | Smoothstep from `z_hover` to `z_grasp` above pick block | 350-353 |
| 0.25-0.42 | Dwell | Hold at `z_grasp` above pick block, `want_grip=True` | 354-357 |
| 0.42-0.55 | Lift | Smoothstep from `z_grasp` to `z_lift` | 358-361 |
| 0.55-0.75 | Carry | Bezier-like path from pick to place via midpoint | 362-374 |
| 0.75-0.88 | Descend (place) | Smoothstep from `z_lift` to `place_z` | 375-378 |
| 0.88-0.92 | Settle | Hold at place position | 379-382 |
| 0.92-0.95 | Release | `want_grip=False`, slight upward retreat | 383-386 |
| 0.95-1.00 | Retreat | Rise to `z_hover` | 387-390 |
| `i > le` | Post-grasp | Fixed at `place_xy` at `z_hover` | 343-345 |

**Critical observation:** Only lines 337-341 (`i < ls`) use the real human wrist trajectory. During the entire grasp window (lines 346-390), the human's actual hand motion is completely ignored. The arm follows hardcoded waypoints (`pick_xy0`, `place_xy0`) with smoothstep interpolation.

**What `p` controls beyond the target position:**
- `want_grip` is derived from `p` thresholds: `True` when `p >= 0.25`, `False` when `p >= 0.92` (lines 349-390)
- Block heights: `z_grasp`, `z_hover`, `z_lift` are hardcoded constants (lines 293-295), not trajectory-derived

### The `want_grip` / Kinematic Attachment Pipeline

**Lines 399-449:** The grasp/release logic is driven by `want_grip`:

1. **ATTACH (lines 402-424):** When `want_grip=True` and `grasped_obj is None`, find the nearest block where `xy_dist < 0.15` and `z_gap < 0.08`. Record `grasp_offset = block_pos - palm_center`.
2. **RELEASE (lines 427-449):** When `want_grip=False` and `grasped_obj is not None`, release the block, pin it at current XY + correct stack Z, start `SETTLE_FRAMES` countdown.
3. **BLEND (lines 481-497):** During substeps, block tracks `palm_center + grasp_offset` with smoothstep blend over `BLEND_FRAMES=12`.

**Key insight:** The attachment/release logic itself is proximity-based and agnostic to how `want_grip` was determined. It does NOT depend on `p`. Once we change `want_grip` to come from `grasping[i]`, the kinematic attachment logic works unchanged.

### Block Positioning and Workspace

**Lines 176-201:** Block positions are overridden to `desired_pick_xy = [0.38, -0.03]` and `desired_support_xy = [0.38, -0.31]` regardless of calibrated object positions. The `table_x_offset = 0.32 - 0.5 = -0.18` adjusts for table position.

Final block positions in sim:
- Pick block: `[0.38 - (-0.18) + (-0.18), -0.03, 0.78 + 0.03 + 0.01] = [0.38, -0.03, 0.82]` -- wait, let me trace this correctly.
- Line 200: `data.qpos[qa:qa+3] = [xy[0] + table_x_offset, xy[1], G1_TABLE_HEIGHT + BLOCK_HALF + 0.01]`
- `xy = obj_xy[pick_nm] = desired_pick_xy - [table_x_offset, 0] = [0.38 - (-0.18), -0.03] = [0.56, -0.03]`
- Position: `[0.56 + (-0.18), -0.03, 0.78 + 0.03 + 0.01] = [0.38, -0.03, 0.82]`

So blocks are at XY `[0.38, -0.03]` and `[0.38, -0.31]` at Z = 0.82 (table height 0.78 + block half 0.03 + margin 0.01).

### IK Solver Analysis

**`ik_solve()` function, lines 89-113:**
- Damped least squares with `0.005 * np.eye(3)` damping
- Max 150 iterations
- Convergence threshold: `err < 0.004` (4mm)
- Null-space bias toward `SEED` with gain `0.05`
- Step size: `min(1.0, 0.25 + norm(err))` -- adaptive, larger steps for larger errors
- Uses wrist body position, compensated for palm-to-wrist offset (`w2p`)
- Previous frame's solution (`last_arm_q`) used as starting configuration -- already warm-started

**Observation:** The IK solver is well-implemented. No changes needed for Phase 2. The only risk is convergence failure when targets move outside the reachable workspace.

### Calibrated Workspace Data (Stack2, Trimmed)

From the actual calibrated data:

| Metric | Value |
|--------|-------|
| Total frames (trimmed) | 229 |
| Wrist X range | [0.187, 0.471] |
| Wrist Y range | [-0.055, 0.259] |
| Wrist Z range | [0.298, 0.459] |
| Grasp frames | 21 |
| First grasp frame | 30 |
| Last grasp frame | 179 |
| Scale factor | 0.214 |
| Table height (G1) | 0.78 |
| Block center Z on table | ~0.82 |

### `ls`/`le` Computation (Lines 181-187)

```python
n = len(wrist)          # 229 (trimmed)
grip_idx = np.where(grasping > 0)[0]   # all grasp frame indices
late = grip_idx[grip_idx >= int(0.6*n)]  # frames >= 137
```

After trimming, `0.6 * 229 = 137`. Grasp frames in trimmed data: 30-179. Frames >= 137 are a subset of the late grasping cluster. This means `ls` will be around frame 137+ and `le` around frame 179+5=184.

**Problem for retargeting:** The `ls`/`le` computation is only used for the `p` variable. When we eliminate `p`, we still need these for initial block selection (lines 189-191 use `wrist[ls:le]` to identify which block is the pick target by proximity). We should keep that computation but remove its use for motion generation.

### `r3d_to_sim_axes()` in `calibrate_workspace.py`

**Lines 31-36:**
```python
def r3d_to_sim_axes(pts):
    # R3D (X_right, Y_up, Z_toward) -> sim (X_fwd, Y_left, Z_up)
    return np.column_stack([-pts[:, 2], -pts[:, 0], pts[:, 1]])
```

This maps:
- R3D X (right) -> sim -Y (left)
- R3D Y (up) -> sim Z (up)
- R3D Z (toward camera) -> sim -X (backward from robot)

For a human reaching forward: R3D Z decreases -> sim X increases (hand moves toward robot workspace). This is correct for forward reach motions.

**RET-04 validation approach:** Render wrist trajectory as colored markers in sim to visually verify axis alignment. The simplest approach is to print trajectory bounds and visually confirm directionality during sim run.

---

## 2. What Must Change

### RET-01: Robot arm follows real wrist trajectory for entire simulation

**Lines to remove:** 324, 337-390 (the `p` computation and the entire `if/elif/else` choreography block).

**Replacement:** For ALL frames (not just `i < ls`):
```python
target = np.array([wrist[i][0], wrist[i][1],
                   max(wrist[i][2], G1_TABLE_HEIGHT + 0.05)])
```

The Z floor changes from `G1_TABLE_HEIGHT + 0.15` (current pre-grasp safety, too high for actual grasping) to `G1_TABLE_HEIGHT + 0.05` (just above table -- allows the hand to reach block-grasp height at `G1_TABLE_HEIGHT + BLOCK_HALF = 0.81`).

**Height analysis:**
- Table surface: Z = 0.78
- Block center: Z = 0.78 + 0.03 = 0.81
- Palm needs to reach block center for grasp: Z ~0.81
- Safety floor: Z = 0.78 + 0.05 = 0.83 -- wait, this is ABOVE block center. Let me reconsider.

Actually, the current IK target includes a palm-to-wrist offset compensation (line 98: `wrist_target = target - w2p`). The `target` is where the palm center should be, and the IK drives the wrist body to achieve that palm position. So `target[2]` should be approximately at block center height.

With `G1_TABLE_HEIGHT = 0.78` and `BLOCK_HALF = 0.03`:
- Block top = 0.78 + 0.06 = 0.84
- Block center = 0.78 + 0.03 = 0.81
- `z_grasp = G1_TABLE_HEIGHT + BLOCK_HALF - 0.005 = 0.805` (current, line 293)

The safety floor should be BELOW grasp height: `G1_TABLE_HEIGHT + 0.02 = 0.80` allows palm to descend to block center. If the human data has the wrist at very low Z values (after calibration), clamping to 0.80 keeps the palm above the table surface while allowing grasp.

**Revised Z floor:** `max(wrist[i][2], G1_TABLE_HEIGHT + 0.02)` -- this is 2cm above table surface, well below block top (6cm), allowing the palm to descend into grasp position.

### RET-02: Grasp triggering from `grasping` signal + proximity

**Remove:** All `want_grip` assignments inside the `p`-based choreography (lines 349, 353, 357, 361, 374, 378, 382, 386, 390).

**Replacement:** A data-driven state machine:

```python
# Grasp intent from tracking data
raw_grip = bool(grasping[i] > 0)

# Require proximity to a block for actual grip activation
if raw_grip and grasped_obj is None:
    # Check if palm is near any block
    pc = palm_center(model, data)
    min_dist = min(np.linalg.norm(data.xpos[bid][:2] - pc[:2]) for bid in obj_body_ids)
    want_grip = (min_dist < 0.15)  # same threshold as attachment logic
else:
    want_grip = raw_grip
```

**Why proximity gating matters:** The tracking `grasping` signal is noisy (34 segments across the full trajectory, many false positives). Without proximity gating, the hand would attempt to grasp whenever the `grasping` signal is True, even when far from any block. The existing attachment logic (line 413) already has a proximity check (`best_dist < 0.15`), but `want_grip` drives finger closure (line 456) -- we don't want fingers closing when the hand is far from blocks.

**State machine approach for grip stability:**

The tracking `grasping` signal has short spurious segments (1-6 frames). If we directly use `grasping[i]`, the hand will rapidly open/close. We need hysteresis:

```python
# Grip state machine with hysteresis
if grip_state == "open":
    if raw_grip and palm_near_block:
        grip_frames_on += 1
        if grip_frames_on >= 3:  # require 3 consecutive frames
            grip_state = "closing"
            want_grip = True
    else:
        grip_frames_on = 0
        want_grip = False
elif grip_state == "closing" or grip_state == "holding":
    if grasped_obj is not None:
        grip_state = "holding"
    if not raw_grip:
        grip_frames_off += 1
        if grip_frames_off >= 3:  # require 3 consecutive release frames
            grip_state = "open"
            want_grip = False
        else:
            want_grip = True  # hold during spurious release
    else:
        grip_frames_off = 0
        want_grip = True
```

Actually, for simplicity and robustness, a simpler approach: use `grasping[i]` directly for `want_grip`, but only when proximity is satisfied. The existing attachment logic (lines 402-424) already requires `best_dist < 0.15` and `z_gap < 0.08`, so spurious `want_grip=True` when far from blocks will just cause finger closure with no attachment -- acceptable visual behavior.

**Simplest correct implementation:**
```python
want_grip = bool(grasping[i] > 0)
```

This works because:
1. Finger closure from false-positive `grasping` signals looks natural (hand opens/closes during motion)
2. Block attachment still requires proximity (line 413: `best_dist < 0.15`)
3. The trimmed trajectory already focuses on the action window, so most `grasping=True` frames are near the block

### RET-03: IK target clamping + convergence failure logging

**Workspace envelope for clamping:**

Based on the current sim setup:
- Robot base at origin (0, 0, 0)
- Table at X=0.32, Y=0, Z=0.78/2 (extends 0.4m in X from center)
- Blocks at approximately X=0.38, Y=-0.03 to -0.31
- G1 right shoulder approximately at X=0.0, Y=-0.15, Z=1.2 (from standing keyframe)
- G1 effective arm reach: ~0.5m from shoulder

**Reachable workspace bounds (conservative):**

| Axis | Min | Max | Rationale |
|------|-----|-----|-----------|
| X | 0.10 | 0.65 | Behind shoulder to full forward reach on table |
| Y | -0.45 | 0.20 | Right workspace (robot's right arm) to center |
| Z | 0.80 | 1.20 | Table surface to above-head reach |

These match the research SUMMARY.md suggestion: X [0.15, 0.65], Y [-0.45, 0.15], Z [table+0.05, table+0.4]. Adjusted slightly for the actual calibrated data range (Y goes up to 0.259 in calibrated data, so upper Y bound should be 0.30 to not clip).

**Revised bounds:**
- X: [0.10, 0.65]
- Y: [-0.45, 0.30]
- Z: [G1_TABLE_HEIGHT + 0.02, G1_TABLE_HEIGHT + 0.42] = [0.80, 1.20]

**Clamping implementation:**
```python
WORKSPACE_MIN = np.array([0.10, -0.45, G1_TABLE_HEIGHT + 0.02])
WORKSPACE_MAX = np.array([0.65,  0.30, G1_TABLE_HEIGHT + 0.42])

target_clamped = np.clip(target, WORKSPACE_MIN, WORKSPACE_MAX)
if not np.array_equal(target, target_clamped):
    n_clamped += 1  # count for logging
target = target_clamped
```

**Convergence failure logging:**

After IK solve, check residual:
```python
pc = palm_center(model, data)
ik_err = np.linalg.norm(target - pc)
if ik_err > 0.02:  # 2cm threshold
    ik_failures.append((i, ik_err))
```

Log summary at end:
```python
if ik_failures:
    print(f"  IK convergence failures: {len(ik_failures)} frames")
    for frame, err in ik_failures[:5]:
        print(f"    Frame {frame}: err={err:.4f}m")
```

### RET-04: Visual coordinate frame validation

**Approach 1 (minimal, recommended):** Print trajectory statistics during sim to verify alignment:
```python
# At start of sim, before frame loop:
print(f"  Trajectory X: [{wrist[:,0].min():.3f}, {wrist[:,0].max():.3f}]")
print(f"  Trajectory Y: [{wrist[:,1].min():.3f}, {wrist[:,1].max():.3f}]")
print(f"  Trajectory Z: [{wrist[:,2].min():.3f}, {wrist[:,2].max():.3f}]")
print(f"  Block pick: {desired_pick_xy}")
print(f"  Block support: {desired_support_xy}")
```

If the wrist trajectory range overlaps with block positions, axes are aligned. If X goes to 0.47 and blocks are at X=0.38, the hand is reaching into the block area.

**Approach 2 (visual overlay):** Add trajectory visualization markers to the sim. This would require adding small sphere geoms at trajectory waypoints. However, this adds complexity for a one-time validation. The statistical approach is sufficient for Phase 2.

**Approach 3 (debug print during run):** Already exists at line 522-524 -- the periodic frame log shows `palm=` position. Verify that when the hand approaches the block, the palm X/Y converge toward block X/Y.

---

## 3. Variables and Constants to Track

### Variables that are REMOVED
- `p` (line 324) -- progress variable, entirely removed
- `z_hover`, `z_lift` (lines 294-295) -- only used by choreography. `z_grasp` (line 293) is still useful for block height reference but not used in new code
- `pick_xy0`, `place_xy0` (lines 283, 288) -- used in choreography for IK targets. Still needed for block selection (line 190) and place Z computation (line 330), but NOT for arm trajectory
- `place_xy`, `place_z`, `palm_z_offset` (lines 327-334) -- only used in choreography targets

### Variables that are KEPT
- `ls`, `le`, `win` (lines 182-187) -- still needed for block selection (which block to pick) and for reference, but NOT for motion generation
- `desired_pick_xy`, `desired_support_xy` (lines 177-178) -- block placement positions
- `pick_nm`, `support_nm` (lines 190-191) -- which block to grasp
- All grasp attachment logic (lines 306-319, 399-449) -- works unchanged
- `finger_ctrl` and blend logic (lines 303, 456-458) -- finger animation unchanged
- `smoothstep()` function (lines 131-133) -- still used in kinematic blend (line 486)

### New variables
- `WORKSPACE_MIN`, `WORKSPACE_MAX` -- IK target clamp bounds (new constants)
- `ik_failures` -- list of (frame, error) for convergence logging
- `n_clamped` -- count of frames where target was clamped

---

## 4. Code Change Map

### Lines to DELETE in `mujoco_g1_v10.py`

| Lines | Content | Reason |
|-------|---------|--------|
| 293-295 | `z_grasp`, `z_hover`, `z_lift` constants | Only used by choreography |
| 324 | `p = 0.0 if (i < ls) else ...` | The `p` variable itself |
| 326-335 | `palm_z_offset`, `place_xy`, `place_z` | Choreography targets |
| 337-390 | Entire `if i < ls: ... elif ... else:` block | The choreographed motion |

### Lines to MODIFY

| Lines | Current | New | Reason |
|-------|---------|-----|--------|
| 522 | `print(f"F{i:03d} p={p:.2f} grip=...")` | `print(f"F{i:03d} grip=...")` | Remove `p` from log |

### Lines to ADD (after line 295, before renderer setup)

1. Workspace clamp constants (2 lines)
2. IK failure tracking variables (2 lines)

### Lines to ADD (replacing lines 324-390)

1. IK target from real trajectory with Z floor and workspace clamping (~6 lines)
2. `want_grip` from `grasping[i]` (1 line)

### Lines to ADD (after IK solve, ~line 397)

1. IK convergence check and logging (~4 lines)

### Lines to ADD (after frame loop, before final stats)

1. Summary of IK failures and clamped frames (~6 lines)

---

## 5. Risk Assessment for Phase 2

### Risk 1: Hand never reaches block because Z floor is too high (CRITICAL)

The current pre-grasp code uses `Z floor = TABLE_HEIGHT + 0.15 = 0.93`. The block top is at 0.84. With this floor, the hand can never descend below 0.93 -- it can never reach the block at 0.81-0.84.

**Mitigation:** Z floor MUST be lowered to `TABLE_HEIGHT + 0.02 = 0.80`, which is below block center (0.81). The calibrated wrist data Z range goes down to 0.298 -- this is well below even the table surface. The clamping will bring it up to 0.80.

### Risk 2: Calibrated wrist trajectory does not pass near blocks (HIGH)

The calibrated wrist X range is [0.187, 0.471] and Y range is [-0.055, 0.259]. The blocks are at X=0.38, Y=-0.03 (pick) and X=0.38, Y=-0.31 (support). The wrist trajectory DOES overlap with the pick block (X=0.38 is within range, Y=-0.03 is within range). But the support block at Y=-0.31 is outside the wrist Y range. This is expected -- the place location is handled by the carry motion, and the hand will carry the block there.

Wait -- with true retargeting, the hand follows the actual human trajectory. If the human carried the block to a different location than the support block, the block will be placed at the wrong position. This is inherent to true retargeting -- the robot does what the human did, not what we want.

**Mitigation:** For Phase 2, accept that the block may not be placed exactly on the support block. The STACKED=True check may fail. This is a known trade-off: we're switching from "always stacks correctly" (choreography) to "faithfully reproduces human motion" (retargeting). If STACKED=True is required, we may need to adjust block positions to match the calibrated trajectory's place location rather than hardcoded positions.

**Alternative:** Keep `desired_pick_xy` and `desired_support_xy` but derive them from the trajectory data:
- `desired_pick_xy` = wrist position at first grasp frame
- `desired_support_xy` = wrist position at last grasp frame (where the block is placed)

This aligns blocks with where the hand actually goes, maximizing the chance of successful grasp and placement.

### Risk 3: IK solver diverges with rapid trajectory motion (MEDIUM)

The IK solver uses the previous frame's solution as the starting configuration (line 393-394). With choreographed motion, targets move smoothly (smoothstep interpolation). With real trajectories, even after smoothing and velocity clamping (3cm/frame max), the IK may struggle with direction changes.

**Mitigation:** The velocity clamp already limits motion to 3cm/frame. At 10 FPS, this is 0.3 m/s. The IK solver with 150 iterations and adaptive step should handle this. Add convergence failure logging (RET-03) to detect problems.

### Risk 4: STACKED=True regression (HIGH)

The choreographed motion was specifically designed to produce STACKED=True. With true retargeting, stacking depends on:
1. The human actually stacking in the video
2. The trajectory calibration placing the wrist near the blocks
3. The grasping signal triggering at the right time
4. The hand carrying the block to the support block location

**Mitigation:** For Phase 2, we accept that STACKED=True may not pass. The success criterion is tracking accuracy (<5cm RMS), not stacking success. If needed, block positions can be adjusted to match the trajectory. This is a philosophical shift: correctness of retargeting is the primary goal, not the specific task outcome.

**However:** We should attempt to make it work by aligning block positions with the trajectory. If the human picked a block at calibrated position A and placed it at calibrated position B, we should place sim blocks at A and B (after the pick/place detection from grasping signal + wrist proximity).

---

## 6. Block Position Alignment Strategy

Instead of hardcoding `desired_pick_xy = [0.38, -0.03]`, derive block positions from the trajectory:

```python
# Find pick position: wrist XY at first grasp frame
grip_idx = np.where(grasping > 0)[0]
if len(grip_idx) > 0:
    pick_xy_from_traj = wrist[grip_idx[0], :2]
    # Find place position: wrist XY at last grasp frame
    place_xy_from_traj = wrist[grip_idx[-1], :2]
```

But this creates a chicken-and-egg problem: we need block positions to set up the sim, but we also need block positions to match where the hand goes.

**Solution:** Keep the current block positioning logic (lines 176-201) that places blocks in the reachable workspace. The trajectory will be close enough (the calibration aligns object centroids to sim workspace center). The key change is that the IK target comes from `wrist[i]` instead of choreographed waypoints -- the hand will naturally approach the block's actual position because the calibration aligned the workspace.

If the hand doesn't reach the block exactly, the proximity threshold (0.15m XY, 0.08m Z) in the attachment logic provides tolerance.

---

## 7. Integration with Phase 1 Outputs

### Trimmed Data Impact

Phase 1 trimmed stack2 from 951 to 229 frames (22.9s). The trimmed window [547, 776) captures the actual stacking action. After trimming:

- `n = 229` frames
- Grasping frames: indices 30-179 within the trimmed array
- `ls`/`le` computation: `late = grip_idx[grip_idx >= int(0.6 * 229)]` = frames >= 137
- The late grasping cluster starts around frame 137-179 of the trimmed array

This is important: with trimming, the "60% late filter" captures the actual stacking action. Without trimming (951 frames), 60% = frame 570, which coincidentally also captures the action. But trimming makes the data cleaner.

### Smoothed Data Quality

Phase 1 applied:
1. Gap-length-aware interpolation (CubicSpline/PCHIP/linear)
2. Savitzky-Golay filter (window=7, polyorder=3)
3. Velocity clamp (3cm/frame max)

The smoothed trajectory should produce smooth IK targets. No additional smoothing needed in the simulation.

---

## 8. Detailed Change Specification

### Step 1: Add workspace clamp constants (after line 43)

```python
# Reachable workspace envelope for IK target clamping
WORKSPACE_MIN = np.array([0.10, -0.45, 0.80])   # X_min, Y_min, Z_min (table + 0.02)
WORKSPACE_MAX = np.array([0.65,  0.30, 1.20])   # X_max, Y_max, Z_max (table + 0.42)
```

### Step 2: Remove `p` choreography, replace with trajectory following (lines 293-390)

Delete lines 293-295 (`z_grasp`, `z_hover`, `z_lift`).

Replace lines 324-390 with:
```python
        # IK target: follow real wrist trajectory (RET-01)
        target = wrist[i].copy()
        target[2] = max(target[2], G1_TABLE_HEIGHT + 0.02)  # Z floor: 2cm above table

        # Clamp to reachable workspace (RET-03)
        target_pre_clamp = target.copy()
        target = np.clip(target, WORKSPACE_MIN, WORKSPACE_MAX)
        if not np.allclose(target, target_pre_clamp, atol=1e-6):
            n_clamped += 1

        # Grasp intent from tracking data (RET-02)
        want_grip = bool(grasping[i] > 0)
```

Also remove lines 326-335 (place target computation that uses choreography variables).

### Step 3: Add IK convergence logging (after line 396)

```python
        # IK convergence check (RET-03)
        pc_check = palm_center(model, data)
        ik_err = np.linalg.norm(target - pc_check)
        if ik_err > 0.02:
            ik_failures.append((i, float(ik_err)))
```

### Step 4: Add summary logging (before final stats, ~line 526)

```python
    # RET-03: Log clamping and convergence stats
    if n_clamped > 0:
        print(f"  Workspace clamping: {n_clamped}/{n} frames clamped to reachable envelope")
    if ik_failures:
        print(f"  IK convergence failures: {len(ik_failures)}/{n} frames (>{0.02}m error)")
        for frame, err in ik_failures[:5]:
            print(f"    Frame {frame:03d}: err={err:.4f}m")
        if len(ik_failures) > 5:
            print(f"    ... and {len(ik_failures)-5} more")
```

### Step 5: Update periodic log to remove `p` (line 522)

Remove `p={p:.2f}` from the print statement since `p` no longer exists.

### Step 6: Add trajectory stats for RET-04 validation (before frame loop)

```python
    # RET-04: Trajectory workspace validation
    print(f"  Trajectory X: [{wrist[:,0].min():.3f}, {wrist[:,0].max():.3f}]")
    print(f"  Trajectory Y: [{wrist[:,1].min():.3f}, {wrist[:,1].max():.3f}]")
    print(f"  Trajectory Z: [{wrist[:,2].min():.3f}, {wrist[:,2].max():.3f}]")
    print(f"  Blocks: pick={desired_pick_xy.round(3)}, support={desired_support_xy.round(3)}")
    # Verify overlap: wrist range should encompass block positions
    wx_range = [wrist[:,0].min(), wrist[:,0].max()]
    wy_range = [wrist[:,1].min(), wrist[:,1].max()]
    pick_in_range = (wx_range[0] <= desired_pick_xy[0] <= wx_range[1] and
                     wy_range[0] <= desired_pick_xy[1] <= wy_range[1])
    print(f"  Pick block within wrist XY range: {pick_in_range}")
```

---

## 9. Verification Plan

### RET-01 Verification: Tracking accuracy

```python
# After sim run, compute RMS tracking error
errors = []
for i in range(n):
    pc = palm_positions[i]  # need to record during sim
    target = wrist[i].copy()
    target[2] = max(target[2], G1_TABLE_HEIGHT + 0.02)
    target = np.clip(target, WORKSPACE_MIN, WORKSPACE_MAX)
    errors.append(np.linalg.norm(pc - target))
rms_err = np.sqrt(np.mean(np.array(errors)**2))
print(f"RMS tracking error: {rms_err:.4f}m")
assert rms_err < 0.05, f"RET-01 FAIL: RMS error {rms_err:.4f} > 0.05m"
```

To record palm positions, add to the frame loop:
```python
palm_positions = []
# ... in frame loop, after IK solve:
palm_positions.append(palm_center(model, data).copy())
```

### RET-02 Verification: No `p` variable in code

```bash
grep -n "^\s*p\s*=" mujoco_g1_v10.py  # should return nothing
grep -n "want_grip.*p\s*>" mujoco_g1_v10.py  # should return nothing
```

### RET-03 Verification: Clamping and logging

Verify that the sim output includes:
- "Workspace clamping: X/N frames clamped" (if any frames were clamped)
- "IK convergence failures: X/N frames" (if any failures occurred)

### RET-04 Verification: Axis alignment

Verify that the sim output includes trajectory range stats, and that the pick block XY falls within the wrist trajectory XY range.

---

*Research complete. Ready for planning.*
