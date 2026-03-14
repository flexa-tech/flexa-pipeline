# Quick Task 9: Fix calibration scale — anchor wrist to objects — Context

**Gathered:** 2026-03-14
**Status:** Ready for planning

<domain>
## Task Boundary

Fix `calibrate_workspace.py` to produce physically grounded calibration instead of arbitrary range mapping. The current scale=0.222 crushes trajectory to 22% of real size. Wrist never reaches block.

</domain>

<decisions>
## Implementation Decisions

### Scale computation
- **Scale=1.0** — R3D depth data is physical meters (Apple LiDAR). No scaling needed. The axis swap (`r3d_to_sim_axes`) handles coordinate frame conversion. Human arm reach exceeds G1 workspace — out-of-bounds frames handled by existing sim workspace clamping (`WORKSPACE_MIN/MAX`).

### Object coordinate handling
- **Flag manual objects as sim coords** — when `source == "manual"` in detection JSON, skip `r3d_to_sim_axes()`. Manual `--objects` values are already in sim space. Only axis-swap detected (GroundingDINO) objects.

### Anchor point
- **Grasp centroid** — average wrist position over ALL grasp frames. NOT onset (onset=frame 0 due to noisy HaMeR grasping signal). Centroid represents where hand spends most time during manipulation, which is near the blocks.

### Claude's Discretion
- Z correction logic (force objects to table height) — keep or adjust as needed
- Workspace clamping in calibration vs relying on sim-side clamping

</decisions>

<specifics>
## Specific Ideas

- The `source: "manual"` flag already exists in `_write_object_detections()` in `run_pipeline.py`
- Offset formula: `offset = obj_centroid_sim - r3d_to_sim_axes(wrist_grasp_centroid)`
- After transform: clamp wrist to workspace bounds or let sim handle it
- Keep the Z correction (force object Z to table height) — still needed

</specifics>
