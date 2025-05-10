import os
import xml.etree.ElementTree as ET
from PIL import Image


# Define class names based on your YOLO classes
class_names = ["drone"]
img_types = ['.JPEG', '.jpg']


def yolo_to_xml(txt_file, img_dir, output_dir):
    # Extract the image filename from the text file name
    base_filename = os.path.splitext(os.path.basename(txt_file))[0]
    # if base_filename != 'classes':
    try:
        img_file = os.path.join(img_dir, base_filename + img_types[0])
    except:
        img_file = os.path.join(img_dir, base_filename + img_types[1])

    # Load the image to get its dimensions
    with Image.open(img_file) as img:
        img_width, img_height = img.size

    # Read the YOLO format text file
    with open(txt_file, 'r') as f:
        lines = f.readlines()

    # Create the XML structure
    root = ET.Element('annotation')
    try:
        ET.SubElement(root, 'filename').text = base_filename + img_types[0]
    except:
        ET.SubElement(root, 'filename').text = base_filename + img_types[1]

    size = ET.SubElement(root, 'size')
    ET.SubElement(size, 'width').text = str(img_width)
    ET.SubElement(size, 'height').text = str(img_height)
    ET.SubElement(size, 'depth').text = '3'

    # Iterate over each line in the text file
    for line in lines:
        parts = line.strip().split()
        class_name = parts[0]
        x_center, y_center, width, height = map(float, parts[1:])

        # Convert YOLO format coordinates back to XML format
        xmin = int((x_center - width / 2) * img_width)
        ymin = int((y_center - height / 2) * img_height)
        xmax = int((x_center + width / 2) * img_width)
        ymax = int((y_center + height / 2) * img_height)

        # Create object element
        obj = ET.SubElement(root, 'object')
        ET.SubElement(obj, 'name').text = class_name
        bndbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(xmin)
        ET.SubElement(bndbox, 'ymin').text = str(ymin)
        ET.SubElement(bndbox, 'xmax').text = str(xmax)
        ET.SubElement(bndbox, 'ymax').text = str(ymax)

    # Write to the XML file
    xml_file = os.path.join(output_dir, base_filename + '.xml')
    tree = ET.ElementTree(root)
    tree.write(xml_file, encoding='utf-8', xml_declaration=True)


def convert_yolo_to_xml(txt_dir, img_dir, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate through text files in the directory
    for txt_file in os.listdir(txt_dir):
        if txt_file.endswith('.txt') and not txt_file.startswith('classes'):
            txt_path = os.path.join(txt_dir, txt_file)
            yolo_to_xml(txt_path, img_dir, output_dir)


# Convert YOLO to XML
convert_yolo_to_xml('./drone/train', './drone/train', './drone/xmltrain')
#convert_yolo_to_xml('./drone/train', './drone/xmltrain')
convert_yolo_to_xml('./drone/test', './drone/test', './drone/xmltest')
#convert_yolo_to_xml('./drone/test', './drone/xmltest')

