"""Convert URDF hand models to MJCF for use with MuJoCo/Mink."""

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def _find_leaf_bodies(worldbody: ET.Element) -> list[str]:
    """Find leaf bodies (no child bodies) in MJCF XML - these are fingertips."""
    leaf_bodies = []

    def _walk(elem):
        child_bodies = elem.findall("body")
        if not child_bodies and elem.tag == "body":
            name = elem.get("name", "")
            if name:
                leaf_bodies.append(name)
        for child in child_bodies:
            _walk(child)

    _walk(worldbody)
    return leaf_bodies


def _compute_fingertip_offsets(model, leaf_body_names: list[str]) -> dict[str, list[float]]:
    """Compute fingertip offsets in local body frame using MuJoCo geom data.

    For each leaf body, find its geoms and compute the max extent in the
    primary axis (typically z for finger extension).
    """
    import mujoco

    offsets = {}
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    for bname in leaf_body_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
        if body_id < 0:
            offsets[bname] = [0.0, 0.0, 0.02]
            continue

        # Find all geoms belonging to this body
        max_extent = 0.0
        geom_center = [0.0, 0.0, 0.0]
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] == body_id:
                gpos = model.geom_pos[gid]
                gsize = model.geom_size[gid]
                # Compute extent: geom center + size in each axis
                for axis in range(3):
                    extent = abs(gpos[axis]) + gsize[min(axis, len(gsize) - 1)]
                    if extent > max_extent:
                        max_extent = extent
                        # Use the axis with maximum extent as the fingertip direction
                        geom_center = [gpos[0], gpos[1], gpos[2]]

        # Find the direction from body origin toward the child body frame.
        # Use the geom center direction, extended by the geom size.
        if max_extent > 0:
            # Use geom center + geom size along the primary axis
            gpos = geom_center
            best_axis = max(range(3), key=lambda a: abs(gpos[a]))
            offset = [0.0, 0.0, 0.0]
            sign = 1.0 if gpos[best_axis] >= 0 else -1.0
            offset[best_axis] = sign * max_extent
            offsets[bname] = offset
        else:
            offsets[bname] = [0.0, 0.0, 0.02]

    return offsets


def _find_all_joints(root: ET.Element) -> list[dict]:
    """Extract all joint elements with their attributes from MJCF XML."""
    joints = []
    for joint in root.iter("joint"):
        name = joint.get("name", "")
        jrange = joint.get("range", "")
        if name and jrange:
            joints.append({"name": name, "range": jrange})
    return joints


def convert_urdf_to_mjcf(
    urdf_path: str,
    output_dir: str,
    hand_name: str | None = None,
) -> str:
    """Convert a URDF file to MJCF with actuators and fingertip sites.

    Args:
        urdf_path: Path to the source URDF file.
        output_dir: Directory to write the output MJCF and meshes.
        hand_name: Optional name for the model. Defaults to URDF filename stem.

    Returns:
        Path to the generated MJCF model.xml file.
    """
    import mujoco

    urdf_path = Path(urdf_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if hand_name is None:
        hand_name = urdf_path.stem

    # Copy mesh files to output
    mesh_src = urdf_path.parent / "meshes"
    mesh_dst = output_dir / "meshes"
    if mesh_src.exists():
        if mesh_dst.exists():
            shutil.rmtree(mesh_dst)
        shutil.copytree(mesh_src, mesh_dst)

    # Load URDF via MuJoCo
    # MuJoCo resolves mesh paths using meshdir. We point meshdir to the
    # meshes/ subdirectory and strip the "meshes/" prefix from filenames.
    urdf_text = urdf_path.read_text()

    mesh_dir_abs = str(urdf_path.parent / "meshes")
    meshdir_tag = f'<mujoco><compiler meshdir="{mesh_dir_abs}/"/></mujoco>'
    if "<mujoco>" not in urdf_text:
        urdf_text = urdf_text.replace("</robot>", f"  {meshdir_tag}\n</robot>")
    # Strip meshes/ prefix so MuJoCo finds files directly in meshdir
    urdf_text = urdf_text.replace('filename="meshes/', 'filename="')

    model = mujoco.MjModel.from_xml_string(urdf_text)
    tmp_xml = output_dir / "_converted.xml"
    mujoco.mj_saveLastXML(str(tmp_xml), model)

    # Post-process: add actuators and fingertip sites
    tree = ET.parse(tmp_xml)
    root = tree.getroot()

    # Fix mesh directory
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("meshdir", "meshes/")

    # Collect joints
    joints = _find_all_joints(root)

    # Add actuators if not present
    actuator_elem = root.find("actuator")
    if actuator_elem is None:
        actuator_elem = ET.SubElement(root, "actuator")
    existing_actuators = {a.get("joint") for a in actuator_elem}
    for joint in joints:
        if joint["name"] not in existing_actuators:
            ET.SubElement(
                actuator_elem,
                "position",
                name=f"act_{joint['name']}",
                joint=joint["name"],
                ctrlrange=joint["range"],
                kp="10",
            )

    # Add fingertip sites on leaf bodies at actual fingertip positions.
    # We compute positions by loading the model and finding geom extents.
    leaf_bodies = _find_leaf_bodies(root.find(".//worldbody"))
    tip_offsets = _compute_fingertip_offsets(model, leaf_bodies)

    if leaf_bodies:
        for body_elem in root.iter("body"):
            bname = body_elem.get("name", "")
            if bname in leaf_bodies:
                existing_sites = [s.get("name") for s in body_elem.findall("site")]
                site_name = f"{bname}_tip"
                if site_name not in existing_sites:
                    offset = tip_offsets.get(bname, [0, 0, 0.02])
                    ET.SubElement(
                        body_elem,
                        "site",
                        name=site_name,
                        pos=f"{offset[0]:.5f} {offset[1]:.5f} {offset[2]:.5f}",
                        size="0.005",
                        rgba="1 0 0 1",
                    )

    # Write final model
    model_path = output_dir / "model.xml"
    tree.write(str(model_path), xml_declaration=True, encoding="unicode")
    tmp_xml.unlink()

    # Validate by loading
    mujoco.MjModel.from_xml_path(str(model_path))

    # Print summary
    print(f"Converted: {urdf_path.name} -> {model_path}")
    print(f"  Joints ({len(joints)}):")
    for j in joints:
        print(f"    {j['name']}  range=[{j['range']}]")
    print(f"  Fingertip sites ({len(leaf_bodies)}):")
    for b in leaf_bodies:
        print(f"    {b}_site (on body '{b}')")

    return str(model_path)
