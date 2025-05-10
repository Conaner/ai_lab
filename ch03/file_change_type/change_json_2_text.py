import os
import json
from PIL import Image

# Define class names based on your YOLO classes
class_names = ["0"]
img_types = ['.JPEG', '.jpg']


def json_to_yolo(json_file, img_dir, output_dir):
    # Load the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)


    # Extract the image filename from the JSON data
    base_filename = os.path.splitext(data['filename'])[0]
    img_file = next((os.path.join(img_dir, base_filename + ext) for ext in img_types if
                     os.path.isfile(os.path.join(img_dir, base_filename + ext))), None)

    if img_file is None:
        raise FileNotFoundError(f"No image file found for {base_filename} with supported types {img_types}")

    # Load the image to get its dimensions
    with Image.open(img_file) as img:
        img_width, img_height = img.size

    # Prepare the YOLO format data
    yolo_data = []

    # Iterate over each object in the JSON file
    for obj in data['objects']:
        print(obj)
        class_id = class_names.index(obj['class_name'])
        # bndbox = obj['objects']

        # Convert standard format coordinates back to YOLO format
        x_center = obj['x_center'] / img_width
        y_center = obj['x_center'] / img_height
        width = obj['width'] / img_width
        height = obj['height'] / img_height

        # Append to YOLO data list
        yolo_data.append(f"{class_id} {x_center} {y_center} {width} {height}")

    # Write to the YOLO format text file
    yolo_file = os.path.join(output_dir, base_filename + '.txt')
    with open(yolo_file, 'w') as f:
        f.write('\n'.join(yolo_data))


def convert_json_to_yolo(json_dir, img_dir, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate through JSON files in the directory
    for json_file in os.listdir(json_dir):
        if json_file.endswith('.json'):
            json_path = os.path.join(json_dir, json_file)
            json_to_yolo(json_path, img_dir, output_dir)


# Convert JSON to YOLO
convert_json_to_yolo(r'.\change_type_dir\json', r'.\train\images', r'.\change_type_dir\txt')
# convert_json_to_yolo('./drone/jsontest', './drone/test', './drone/json2txt_test')
