import os
import json
import xml.etree.ElementTree as ET


def xml_to_json(xml_file, output_dir):
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Extract relevant information
    filename = root.find('filename').text
    img_width = int(root.find('size').find('width').text)
    img_height = int(root.find('size').find('height').text)

    # Prepare a list to hold all object data
    objects_list = []

    # Iterate over each object in the XML file
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        xmin = int(obj.find('bndbox').find('xmin').text)
        ymin = int(obj.find('bndbox').find('ymin').text)
        xmax = int(obj.find('bndbox').find('xmax').text)
        ymax = int(obj.find('bndbox').find('ymax').text)

        # Calculate YOLO format coordinates
        x_center = (xmin + xmax) / (2 * img_width)
        y_center = (ymin + ymax) / (2 * img_height)
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        # Append object data to the list
        objects_list.append({
            "class_name": class_name,
            "x_center": x_center,
            "y_center": y_center,
            "width": width,
            "height": height
        })

    # Construct the JSON data
    json_data = {
        "filename": filename,
        "width": img_width,
        "height": img_height,
        "objects": objects_list
    }

    # Open the JSON file for writing
    json_file = os.path.join(output_dir, os.path.splitext(filename)[0] + '.json')
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=4)


def convert_xml_to_json(xml_dir, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate through XML files in the directory
    for xml_file in os.listdir(xml_dir):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(xml_dir, xml_file)
            xml_to_json(xml_path, output_dir)


# Create output directories
os.makedirs('drone/jsontrain', exist_ok=True)
os.makedirs('drone/jsontest', exist_ok=True)

# Convert XML files to JSON
convert_xml_to_json(r'.\train\xml', r'.\change_type_dir\json')
# convert_xml_to_json('drone/xmltest', 'drone/jsontest')
