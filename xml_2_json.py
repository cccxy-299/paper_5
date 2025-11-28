import xml.etree.ElementTree as ET
import json
import os

def xml_to_coco(xml_folder, output_json_path):
    # Initialize COCO format dictionary
    list_videos = os.listdir(r"G:\dataset\paper5\test")
    coco_format = {
        "info": {
            "description": "Car Accident Dataset",
            "version": "1.0",
            "year": 2023,
            "contributor": "Your Name",
            "date_created": "2023-10-01"
        },
        "licenses": [
            {
                "id": 1,
                "name": "CC BY 4.0",
                "url": "http://creativecommons.org/licenses/by/4.0/"
            }
        ],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "crash",
                "supercategory": "accident"
            }
        ]
    }

    # Initialize IDs
    image_id = 1
    annotation_id = 1

    # Iterate over all XML files in the folder
    for xml_file in os.listdir(xml_folder):
        if not xml_file.endswith('.xml'):
            continue

        xml_path = os.path.join(xml_folder, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        video_index = root.find('filename').text.split("_")[0]
        if video_index not in list_videos:
            continue
        # Extract image information
        image_info = {
            "id": image_id,
            "file_name": root.find('filename').text,
            "width": int(root.find('size/width').text),
            "height": int(root.find('size/height').text),
            "date_captured": "2023-10-01",
            "license": 1,
            "coco_url": "",
            "flickr_url": ""
        }
        coco_format["images"].append(image_info)

        # Extract annotation information
        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            width = xmax - xmin
            height = ymax - ymin

            annotation_info = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [xmin, ymin, width, height],
                "area": width * height,
                "iscrowd": 0,
                "segmentation": []
            }
            coco_format["annotations"].append(annotation_info)
            annotation_id += 1

        # Increment image ID for the next image
        image_id += 1

    # Save the COCO format dictionary to a JSON file
    with open(output_json_path, 'w') as json_file:
        json.dump(coco_format, json_file, indent=4)

# Example usage
xml_folder = r'G:\dataset\paper5\object_detection\xml'  # Replace with the path to your folder containing XML files
output_json_path = 'G:\dataset\paper5\object_detection\instances_val2017.json'  # Replace with the desired output JSON file path
xml_to_coco(xml_folder, output_json_path)