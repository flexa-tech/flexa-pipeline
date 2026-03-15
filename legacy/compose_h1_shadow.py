#!/usr/bin/env python3
"""Compose H1 + Shadow Hand model and return compiled MjModel.

Usage:
    from compose_h1_shadow import build_model
    model = build_model()  # returns compiled MjModel with table + blocks
"""

import mujoco
import numpy as np
from pathlib import Path

from pipeline_config import H1_DIR, SHADOW_DIR

BLOCK_HALF = 0.04
RENDER_W, RENDER_H = 640, 480
TIMESTEP = 0.002

def find_body(root, name):
    for b in root.bodies:
        if b.name == name:
            return b
        found = find_body(b, name)
        if found:
            return found
    return None


def build_model(table_height=0.75, obj_names=None, obj_positions=None):
    """Build composed H1 + Shadow Hand model with table and blocks.
    
    Args:
        table_height: height of table surface
        obj_names: list of object names (default: ["block_a", "block_b"])
        obj_positions: dict of name -> [x, y] positions on table
    
    Returns:
        compiled MjModel
    """
    if obj_names is None:
        obj_names = ["block_a", "block_b"]
    
    h1 = mujoco.MjSpec.from_file(str(H1_DIR / "h1.xml"))
    shadow = mujoco.MjSpec.from_file(str(SHADOW_DIR / "right_hand.xml"))
    
    # Fix naming conflicts
    for m in shadow.materials:
        if m.name == 'black':
            m.name = 'sh_black'
        elif m.name == 'white':
            m.name = 'sh_white'
    
    # Attach shadow hand to H1 right elbow
    elbow = find_body(h1.worldbody, 'right_elbow_link')
    frame = elbow.add_frame()
    frame.pos = np.array([0.28, 0.0, -0.015])
    # Rotate so shadow hand forearm (Z-up) aligns with H1 forearm direction (X-out)
    frame.quat = np.array([0.707107, 0.0, -0.707107, 0.0])
    
    sh_forearm = find_body(shadow.worldbody, 'rh_forearm')
    frame.attach_body(sh_forearm)
    
    # Set options
    h1.option.timestep = TIMESTEP
    h1.option.gravity = np.array([0, 0, -9.81])
    
    # Add visual settings
    h1.visual.global_.offwidth = RENDER_W
    h1.visual.global_.offheight = RENDER_H
    
    # Add skybox texture
    tex = h1.add_texture()
    tex.name = "skybox"
    tex.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    tex.rgb1 = np.array([0.3, 0.5, 0.7])
    tex.rgb2 = np.array([0.0, 0.0, 0.0])
    tex.width = 512
    tex.height = 3072
    
    # Ground plane
    floor = h1.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = np.array([0.0, 0.0, 0.05])
    floor.contype = 1
    floor.conaffinity = 1
    floor.rgba = np.array([0.2, 0.25, 0.3, 1.0])
    
    # Light
    light = h1.worldbody.add_light()
    light.pos = np.array([0.0, 0.0, 2.5])
    light.dir = np.array([0.0, 0.0, -1.0])
    # light.directional = True  # not available via MjSpec, use default
    
    # Table
    table = h1.worldbody.add_body()
    table.name = "table"
    table.pos = np.array([0.55, 0.0, table_height / 2])
    tg = table.add_geom()
    tg.type = mujoco.mjtGeom.mjGEOM_BOX
    tg.size = np.array([0.4, 0.6, table_height / 2])
    tg.rgba = np.array([0.55, 0.35, 0.18, 1.0])
    tg.mass = 50.0
    tg.friction = np.array([1.0, 0.005, 0.0001])
    tg.contype = 1
    tg.conaffinity = 1
    
    # Blocks
    colors = {"block_a": [0.95, 0.15, 0.15, 1.0], "block_b": [0.15, 0.25, 0.95, 1.0]}
    for nm in obj_names:
        block = h1.worldbody.add_body()
        block.name = nm
        bj = block.add_freejoint()
        bj.name = f"{nm}_jnt"
        bg = block.add_geom()
        bg.type = mujoco.mjtGeom.mjGEOM_BOX
        bg.size = np.array([BLOCK_HALF, BLOCK_HALF, BLOCK_HALF])
        bg.rgba = np.array(colors.get(nm, [0.2, 0.8, 0.3, 1.0]))
        bg.mass = 0.05
        bg.friction = np.array([2.0, 0.01, 0.001])
        bg.contype = 1
        bg.conaffinity = 1
    
    # Camera
    cam = h1.worldbody.add_camera()
    cam.name = "front"
    cam.pos = np.array([1.8, 1.0, 1.3])
    cam.fovy = 50.0
    
    # Compile
    model = h1.compile()
    return model


if __name__ == "__main__":
    model = build_model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    
    print(f"Model: {model.nq} qpos, {model.nv} dof, {model.nu} actuators, {model.nbody} bodies")
    
    # Key body positions
    for bname in ["pelvis", "right_elbow_link", "rh_forearm", "rh_palm", 
                   "rh_ffdistal", "rh_thdistal", "block_a", "block_b"]:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
        if bid >= 0:
            print(f"  {bname}: {data.xpos[bid].round(4)}")
    
    # List right arm + hand joints
    print("\nRight arm joints:")
    for i in range(model.njnt):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if n and ("right_shoulder" in n or "right_elbow" in n):
            print(f"  {n}: qa={model.jnt_qposadr[i]} da={model.jnt_dofadr[i]}")
    
    print("\nShadow hand joints:")
    for i in range(model.njnt):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if n and n.startswith("rh_"):
            print(f"  {n}: qa={model.jnt_qposadr[i]} da={model.jnt_dofadr[i]}")
    
    print("\nActuators:")
    for i in range(model.nu):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"  [{i}] {n}")
    
    # H1 standing height
    pelvis_z = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")][2]
    palm_z = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rh_palm")][2]
    print(f"\nPelvis height: {pelvis_z:.3f}")
    print(f"Palm height: {palm_z:.3f}")
    print(f"Suggested table height: {palm_z - 0.10:.3f} to {palm_z + 0.05:.3f}")
