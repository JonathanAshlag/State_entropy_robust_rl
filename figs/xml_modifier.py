import xml.etree.ElementTree as ET
import os

def modify_joint(xml_path, output_dir, joint_name, damping_value=20.0):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for joint in root.iter('joint'):
        if joint.attrib.get('name') == joint_name:
            joint.set('damping', str(damping_value))
            print(f"Hampering joint: {joint_name}")
            break

    fname = f"{os.path.splitext(os.path.basename(xml_path))[0]}_hampered_{joint_name}_damping_{damping_value}.xml"
    out_path = os.path.join(output_dir, fname)
    tree.write(out_path)
    return out_path

def get_all_joint_names(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    return [j.attrib['name'] for j in root.iter('joint') if 'name' in j.attrib and 'joint' in j.attrib['name']]

if __name__ == "__main__":
    original_xml = "mujoco_local/pusher_v5.xml"
    output_dir = "mujoco_local/Pusher_modified"
    os.makedirs(output_dir, exist_ok=True)
    for damping in [0.5]:
        joint_names = get_all_joint_names(original_xml)
        for joint in joint_names:
            if joint.startswith("r_"):  # Only arm joints
                modify_joint(original_xml, output_dir, joint, damping_value=damping)

