import os
import argparse
import json
import cv2
from tqdm import tqdm 
import numpy as np
from pycocotools import mask as mask_utils

# Helper function to generate unique colors for each category
def get_unique_color(category_id):
    np.random.seed(category_id)  # Seed by category_id to ensure unique colors
    color = np.random.randint(0, 255, size=3)
    return tuple(int(c) for c in color)

def create_coco_rgb_masks(image_dir, annotation_file, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)
    
    annotations_by_image = {}
    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(annotation)

    images_by_id = {img['id']: img['file_name'] for img in coco_data['images']}
    
    # Create RGB masks for each image
    for image_id, annotations in tqdm(annotations_by_image.items()):
        image_filename = images_by_id[image_id]
        image_path = os.path.join(image_dir, image_filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Image {image_filename} not found.")
            continue
        height, width, _ = image.shape

        # Initialize an RGB mask with zeros (background by default)
        rgb_mask = np.zeros((height, width, 3), dtype=np.uint8)

        for annotation in annotations:
            category_id = annotation['category_id']
            color = get_unique_color(category_id)  # Assign a unique color for each category
            
            if isinstance(annotation['segmentation'], list):
                # Handle segmentation polygons
                for segmentation in annotation['segmentation']:
                    polygon = np.array(segmentation, dtype=np.int32).reshape(-1, 2)
                    cv2.fillPoly(rgb_mask, [polygon], color)  # Fill polygon with color
            
            elif isinstance(annotation['segmentation'], dict):
                # Handle RLE masks
                rle = mask_utils.frPyObjects(annotation['segmentation'], height, width)
                binary_mask = mask_utils.decode(rle)
                rgb_mask[binary_mask == 1] = color
        
        # Save the RGB mask as an image
        mask_output_path = os.path.join(output_dir, image_filename)
        cv2.imwrite(mask_output_path, rgb_mask)


if __name__=="__main__":
    dir = os.getcwd()

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image_dir", required=True, help="Path to the image directory")
    ap.add_argument('-a', '--annotation', required=True, help="Path to the annotation directory or file")
    ap.add_argument('-o', '--output_dir', default=os.path.join(dir, "Outputs"), help="Path to the output directory")
    args = vars(ap.parse_args())

    image_dir = args["image_dir"]
    annotation = args["annotation"]
    output_dir = args["output_dir"]

    create_coco_rgb_masks(image_dir, annotation, output_dir)
