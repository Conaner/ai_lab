import os
import json
import xml.etree.ElementTree as ET
from xml.dom import minidom

def prettify(elem):
    """Return a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def json_to_xml(json_file, output_dir):
    # Read the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Create the root element
    annotation = ET.Element('annotation')

    # Create and append filename element
    filename = ET.SubElement(annotation, 'filename')
    filename.text = data['filename']

    # Create and append size element
    size = ET.SubElement(annotation, 'size')
    width = ET.SubElement(size, 'width')
    width.text = str(data['width'])
    height = ET.SubElement(size, 'height')
    height.text = str(data['height'])
    
    # Iterate over each object in the JSON file
    for obj in data['objects']:
        object_elem = ET.SubElement(annotation, 'object')
        name = ET.SubElement(object_elem, 'name')
        name.text = obj['class_name']
        
        bndbox = ET.SubElement(object_elem, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        ymin = ET.SubElement(bndbox, 'ymin')
        xmax = ET.SubElement(bndbox, 'xmax')
        ymax = ET.SubElement(bndbox, 'ymax')
        
        # Calculate original coordinates
        x_center = obj['x_center']
        y_center = obj['y_center']
        width = obj['width']
        height = obj['height']
        
        img_width = data['width']
        img_height = data['height']
        
        xmin_val = int((x_center - width / 2) * img_width)
        ymin_val = int((y_center - height / 2) * img_height)
        xmax_val = int((x_center + width / 2) * img_width)
        ymax_val = int((y_center + height / 2) * img_height)
        
        xmin.text = str(xmin_val)
        ymin.text = str(ymin_val)
        xmax.text = str(xmax_val)
        ymax.text = str(ymax_val)

    # Create the XML tree
    tree = ET.ElementTree(annotation)
    
    # Prettify the XML output
    xml_str = prettify(annotation)
    
    # Write to the XML file
    xml_file = os.path.join(output_dir, os.path.splitext(data['filename'])[0] + '.xml')
    with open(xml_file, 'w') as f:
        f.write(xml_str)

def convert_json_to_xml(json_dir, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate through JSON files in the directory
    for json_file in os.listdir(json_dir):
        if json_file.endswith('.json'):
            json_path = os.path.join(json_dir, json_file)
            json_to_xml(json_path, output_dir)

# Create output directories
os.makedirs('appledataset/jsontrain', exist_ok=True)
os.makedirs('appledataset/jsontest', exist_ok=True)

# Convert JSON files to XML
convert_json_to_xml('D:/pythonpa/ch02/appledataset/jsontrain', 'D:/pythonpa/ch02/appledataset/json-xmltrain')
convert_json_to_xml('D:/pythonpa/ch02/appledataset/jsontest', 'D:/pythonpa/ch02/appledataset/json-xmltest')
