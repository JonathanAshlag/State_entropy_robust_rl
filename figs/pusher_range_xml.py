import xml.etree.ElementTree as ET
import os
import shutil

def reduce_joint_ranges(xml_path, output_dir, delta_fraction=0.1,shrink_type="center"):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    os.makedirs(output_dir, exist_ok=True)

    joints = root.findall(".//joint")
    joint_info = []

    for joint in joints:
        if joint.attrib.get("type") != "hinge":
            continue
        name = joint.attrib["name"]
        if "range" not in joint.attrib:
            continue

        # Parse original range
        min_range, max_range = map(float, joint.attrib["range"].split())
        full_range = max_range - min_range
        delta = full_range * delta_fraction

        # Create new range (centered shrink)
        if shrink_type == "center":
            new_min = min_range + delta / 2
            new_max = max_range - delta / 2

        # Create new range (left shrink)
        elif shrink_type == "left":
            new_min = min_range + delta
            new_max = max_range
        # Create new range (right shrink)
        elif shrink_type == "right":
            new_min = min_range
            new_max = max_range - delta

        joint.attrib["range"] = f"{new_min:.5f} {new_max:.5f}"

        joint_info.append((name, min_range, max_range, new_min, new_max))

    # Save modified XML
    modified_xml_path = os.path.join(output_dir, f"reduced_{int(delta_fraction*100)}p_{shrink_type}.xml")
    tree.write(modified_xml_path)
    print(f"Saved modified XML to: {modified_xml_path}")
    return joint_info

# Example usage

xml_file = "mujoco_local/pusher_v5.xml"
output_dir = "mujoco_local/pusher_modified"
for delata in [0,0.05,0.1,0.15, 0.2]:
    for shrink_type in ["left", "right", "center"]:      
        delta_fraction = delata  # 10% reduction
        joint_changes = reduce_joint_ranges(xml_file, output_dir, delta_fraction,shrink_type=shrink_type)
        # for info in joint_changes:
        #     print(f"{info[0]}: {info[1]:.3f}→{info[3]:.3f}, {info[2]:.3f}→{info[4]:.3f}")
