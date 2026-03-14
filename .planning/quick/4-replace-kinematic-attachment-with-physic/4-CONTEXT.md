# Quick Task 4: Replace kinematic attachment with physics-based grasping - Context

**Gathered:** 2026-03-14
**Status:** Ready for planning

<domain>
## Task Boundary

Replace the kinematic block attachment system (grasped_jnt_adr teleportation) in mujoco_g1_v10.py with contact-based physics grasping where fingers physically close around the block and friction holds it.

</domain>

<decisions>
## Implementation Decisions

### Grasp method
- Pure contact friction. No weld constraints, no kinematic cheats. Fingers close, friction holds block.

### Block placement (root cause fix)
- Blocks MUST be placed where the hand actually goes (trajectory-derived positions), not at hardcoded --objects positions. The 0.115m IK error comes from blocks placed at the edge of the workspace — the hand can't reach them. In the human video the hand reaches the block fine; if it doesn't in sim, the block positions are wrong, not the hand.
- The trajectory already has grasp-onset wrist positions from debounce_grasping(). Use those to position blocks.

### Block stability
- Ideally both blocks fully dynamic. Claude's discretion: start with support pinned for stability, switch to fully dynamic if it works.

### Failure mode
- Let it fail visibly. Block drops → simulation continues → STACKED=False. Honest signal that trajectory needs work.

</decisions>

<specifics>
## Specific Ideas

- Remove the grasped_jnt_adr system entirely (lines ~483-614)
- Remove collision exclusion hack (contype/conaffinity toggling)
- Remove post-release settle pin (placed_jnt_adr system)
- Keep finger pre-shaping (distance-based curl) — it helps establish contact
- Tune MuJoCo friction coefficients on block + finger geoms for reliable grip
- May need to increase SUBSTEPS for stable contact dynamics

</specifics>
