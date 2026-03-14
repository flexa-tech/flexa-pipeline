#!/usr/bin/env python3
"""Convert MuJoCo MJCF XML to URDF for Pinocchio.

Extracts kinematic tree, inertials, joint limits from a loaded MuJoCo model
and writes a minimal URDF suitable for Pinocchio IK/dynamics.

Meshes are referenced but not required for kinematic-only usage.
"""

import mujoco
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
from scipy.spatial.transform import Rotation


def quat_wxyz_to_rpy(quat_wxyz):
    """Convert MuJoCo quaternion (w,x,y,z) to URDF RPY (roll,pitch,yaw)."""
    w, x, y, z = quat_wxyz
    r = Rotation.from_quat([x, y, z, w])  # scipy uses (x,y,z,w)
    return r.as_euler('xyz')


def fmt(arr, precision=6):
    """Format array as space-separated string."""
    return ' '.join(f'{v:.{precision}f}' for v in arr)


def mjcf_to_urdf(mjcf_path: str, urdf_path: str, mesh_dir: str = None):
    """Convert MJCF to URDF.

    Args:
        mjcf_path: Path to MJCF XML file
        urdf_path: Output URDF path
        mesh_dir: Relative path to mesh directory (for URDF mesh references)
    """
    model = mujoco.MjModel.from_xml_path(mjcf_path)

    # Build body -> joint mapping
    body_joint = {}  # body_id -> list of joint_ids
    for j in range(model.njnt):
        bid = model.jnt_bodyid[j]
        body_joint.setdefault(bid, []).append(j)

    # MuJoCo joint type mapping
    JOINT_AXES = {0: None, 1: None, 2: None, 3: 'revolute'}  # 0=free, 1=ball, 2=slide, 3=hinge

    root = ET.Element('robot', name='g1')

    # Create links and joints
    for bid in range(model.nbody):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if bname is None:
            bname = f'body_{bid}'

        # Skip world body as parent (handled separately)
        if bid == 0:
            # Create world link
            link = ET.SubElement(root, 'link', name='world')
            continue

        # Create link
        link = ET.SubElement(root, 'link', name=bname)

        mass = model.body_mass[bid]
        if mass > 0:
            inertial = ET.SubElement(link, 'inertial')

            # Inertial frame origin (relative to link frame)
            ipos = model.body_ipos[bid]
            iquat = model.body_iquat[bid]
            irpy = quat_wxyz_to_rpy(iquat)
            ET.SubElement(inertial, 'origin', xyz=fmt(ipos), rpy=fmt(irpy))

            ET.SubElement(inertial, 'mass', value=f'{mass:.6f}')

            # Diagonal inertia (MuJoCo stores principal axes inertia)
            ix, iy, iz = model.body_inertia[bid]
            ET.SubElement(inertial, 'inertia',
                         ixx=f'{ix:.8f}', ixy='0', ixz='0',
                         iyy=f'{iy:.8f}', iyz='0', izz=f'{iz:.8f}')

        # Create joint connecting parent to this body
        parent_id = model.body_parentid[bid]
        parent_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id)
        if parent_name is None:
            parent_name = 'world'

        # Body pose relative to parent
        pos = model.body_pos[bid]
        quat = model.body_quat[bid]
        rpy = quat_wxyz_to_rpy(quat)

        joints_for_body = body_joint.get(bid, [])

        if not joints_for_body:
            # Fixed joint
            joint = ET.SubElement(root, 'joint', name=f'{bname}_fixed', type='fixed')
            ET.SubElement(joint, 'parent', link=parent_name)
            ET.SubElement(joint, 'child', link=bname)
            ET.SubElement(joint, 'origin', xyz=fmt(pos), rpy=fmt(rpy))
        else:
            for jid in joints_for_body:
                jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
                jtype = model.jnt_type[jid]

                if jtype == 0:  # free joint
                    # For floating base, create a fixed joint to world
                    # (Pinocchio handles floating base differently)
                    joint = ET.SubElement(root, 'joint', name=jname, type='floating')
                    ET.SubElement(joint, 'parent', link=parent_name)
                    ET.SubElement(joint, 'child', link=bname)
                    ET.SubElement(joint, 'origin', xyz=fmt(pos), rpy=fmt(rpy))
                elif jtype == 3:  # hinge/revolute
                    joint = ET.SubElement(root, 'joint', name=jname, type='revolute')
                    ET.SubElement(joint, 'parent', link=parent_name)
                    ET.SubElement(joint, 'child', link=bname)
                    ET.SubElement(joint, 'origin', xyz=fmt(pos), rpy=fmt(rpy))

                    axis = model.jnt_axis[jid]
                    ET.SubElement(joint, 'axis', xyz=fmt(axis, 1))

                    lo, hi = model.jnt_range[jid]
                    ET.SubElement(joint, 'limit', lower=f'{lo:.6f}', upper=f'{hi:.6f}',
                                 effort='25.0', velocity='10.0')

    # Pretty-print
    rough_string = ET.tostring(root, encoding='unicode')
    reparsed = minidom.parseString(rough_string)
    pretty = reparsed.toprettyxml(indent='  ')

    # Remove extra XML declaration
    lines = pretty.split('\n')
    if lines[0].startswith('<?xml'):
        lines[0] = '<?xml version="1.0" ?>'

    Path(urdf_path).write_text('\n'.join(lines))
    print(f'Written URDF to {urdf_path}')
    print(f'  Bodies: {model.nbody}, Joints: {model.njnt}')

    return urdf_path


if __name__ == '__main__':
    import sys
    mjcf = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent.parent.parent / 'mujoco_menagerie/unitree_g1/g1_with_hands.xml')
    urdf = sys.argv[2] if len(sys.argv) > 2 else str(Path(__file__).parent.parent / 'models/unitree_g1/g1.urdf')
    mjcf_to_urdf(mjcf, urdf)
