import base64
import io
import os

import cv2
import numpy as np
import pytesseract
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

current_script_dir = os.path.dirname(__file__)
checkpoint_path = os.path.join(current_script_dir, "./../../sam_vit_b_01ec64.pth")
sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
# sam = sam_model_registry["vit_b"](checkpoint="./../../sam_vit_b_01ec64.pth")
mask_generator = SamAutomaticMaskGenerator(sam)


# def segment_and_text_detection(image_data_base64):
#     if image_data_base64.startswith("data:image"):
#         base64_str = image_data_base64.split(",", 1)[1]
#     else:
#         base64_str = image_data_base64

#     image_data = base64.b64decode(base64_str)
#     image = Image.open(io.BytesIO(image_data))
#     test_image = np.array(image)
#     masks = mask_generator.generate(test_image)

#     bounding_boxes = extract_bounding_boxes_and_text_detection(test_image, masks)
#     return bounding_boxes


# def extract_bounding_boxes_and_text_detection(image, masks):
#     results = []
#     for mask in masks:
#         bbox = mask["bbox"]
#         segment = image[bbox[1] : bbox[3], bbox[0] : bbox[2]]

#         if segment.size > 0:
#             contains_text = detect_text_in_segment(segment)
#         else:
#             contains_text = False

#         results.append({"bbox": bbox, "containsText": contains_text})
#     return results


# def detect_text_in_segment(segment):
#     segment_rgb = cv2.cvtColor(segment, cv2.COLOR_BGR2RGB)
#     boxes = pytesseract.image_to_boxes(Image.fromarray(segment_rgb))
#     return len(boxes) > 0  # Returns True if text is detected, False otherwise


def segment_image(image_data_base64):
    # Check and strip the prefix if it's present
    if image_data_base64.startswith("data:image"):
        # Find the comma and get only the base64 part
        base64_str = image_data_base64.split(",", 1)[1]
    else:
        base64_str = image_data_base64

    # Decode the image from base64
    image_data = base64.b64decode(base64_str)

    image = Image.open(io.BytesIO(image_data))
    test_image = np.array(image)

    # Perform segmentation
    masks = mask_generator.generate(test_image)

    # Extract bounding boxes from the masks
    bounding_boxes = extract_bounding_boxes(masks)

    return bounding_boxes


def extract_bounding_boxes(masks):
    bounding_boxes = []
    for mask in enumerate(masks):
        bounding_boxes.append(mask[1]["bbox"])
    return bounding_boxes
