import json
import os
from tkinter import Tk
from tkinter.filedialog import askdirectory

def txt_to_dict(txt_content):
    """Convert a TXT file content (key-value pairs) to a dictionary."""
    data = {}
    for line in txt_content.splitlines():
        if ":" in line:  # Split key-value pairs by the first occurrence of ":"
            key, value = line.split(":", 1)
            data[key.strip()] = value.strip()
    return data

def txt_to_json(txt_content):
    """Convert a TXT file content to a JSON string."""
    # Convert the TXT content to a dictionary
    txt_dict = txt_to_dict(txt_content)

    # Convert the dictionary to a JSON string
    json_string = json.dumps(txt_dict, indent=4)
    return json_string

def select_folder():
    """Open a folder dialog to select a folder containing TXT files."""
    Tk().withdraw()  # Hide the root window
    folder = askdirectory(title="Select a folder containing TXT files")
    return folder

def save_json(json_string, output_folder, input_filename):
    """Save the JSON string to a file in the output folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_filename = os.path.join(output_folder, os.path.basename(input_filename).replace(".txt", ".json"))
    with open(output_filename, "w", encoding="utf-8") as json_file:
        json_file.write(json_string)
    print(f"JSON file saved to: {output_filename}")

def process_txt_folder(input_folder, output_folder):
    """Process all TXT files in the input folder and save JSON files in the output folder."""
    if not os.path.exists(input_folder):
        print(f"The folder '{input_folder}' does not exist. Exiting.")
        return

    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            txt_file_path = os.path.join(input_folder, filename)

            # Read the TXT file
            with open(txt_file_path, "r", encoding="utf-8") as file:
                txt_content = file.read()

            # Convert TXT to JSON
            json_string = txt_to_json(txt_content)

            # Save the JSON to the output folder
            save_json(json_string, output_folder, txt_file_path)

def main():
    # Prompt the user to select a folder containing TXT files
    input_folder = select_folder()
    if not input_folder:
        print("No folder selected. Exiting.")
        return

    # Set the output folder
    output_folder = os.path.join(input_folder, "output")

    # Process all TXT files in the folder
    process_txt_folder(input_folder, output_folder)

if __name__ == "__main__":
    main()