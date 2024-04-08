import base64
import io
import os

import cv2
import numpy as np
import pytesseract

# import tensorflow as tf
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

current_script_dir = os.path.dirname(__file__)
checkpoint_path = os.path.join(current_script_dir, "./../models/sam_vit_b_01ec64.pth")
sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
# sam = sam_model_registry["vit_b"](checkpoint="./../../sam_vit_b_01ec64.pth")
mask_generator = SamAutomaticMaskGenerator(sam)

# Load your trained model (adjust path as necessary)
# model_path = "./../models/obstacle_classifier.h5"
# classification_model = tf.keras.models.load_model(model_path)

# testing

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


# Test

## Previous route functions
# def segment_image(image_data_base64):
#     # Check and strip the prefix if it's present
#     if image_data_base64.startswith("data:image"):
#         # Find the comma and get only the base64 part
#         base64_str = image_data_base64.split(",", 1)[1]
#     else:
#         base64_str = image_data_base64

#     # Decode the image from base64
#     image_data = base64.b64decode(base64_str)

#     image = Image.open(io.BytesIO(image_data))
#     test_image = np.array(image)

#     # Perform segmentation
#     masks = mask_generator.generate(test_image)

#     # Extract bounding boxes from the masks
#     bounding_boxes = extract_bounding_boxes(masks)

#     return bounding_boxes


# def extract_bounding_boxes(masks):
#     bounding_boxes = []
#     for mask in enumerate(masks):
#         bounding_boxes.append(mask[1]["bbox"])
#     return bounding_boxes
## Previous route functions end


# Assuming mask_generator is already defined and loaded as per your previous setup


def classify_segment(segment):
    # Adjust this function based on your model's requirements
    resized_segment = cv2.resize(segment, (250, 250))
    segment_batch = np.expand_dims(resized_segment, axis=0)
    # logits = classification_model.predict(segment_batch)
    # probs = tf.sigmoid(logits).numpy()
    # label = (probs < 0.5).astype(int)
    class_label = "obstacle" if True == 0 else "irrelevant"
    return class_label


def draw_text_boxes_on_segment(segment):
    # This function now returns if text was detected, along with the segment
    segment_rgb = cv2.cvtColor(segment, cv2.COLOR_BGR2RGB)
    boxes = pytesseract.image_to_boxes(Image.fromarray(segment_rgb))
    has_text = len(boxes) > 0
    return has_text


def segment_and_text_detection(image_data_base64):
    # Decode and segment the image
    if image_data_base64.startswith("data:image"):
        base64_str = image_data_base64.split(",", 1)[1]
    else:
        base64_str = image_data_base64

    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data))
    test_image = np.array(image)
    masks = mask_generator.generate(test_image)

    # Process each segment
    results = []
    for mask in masks:
        bbox = [int(coord) for coord in mask["bbox"]]
        segment = test_image[bbox[1] : bbox[3], bbox[0] : bbox[2]]
        if segment.size > 0:
            has_text = draw_text_boxes_on_segment(segment)
            class_label = classify_segment(segment)
            results.append({"bbox": bbox, "hasText": has_text, "type": class_label})
    return results
