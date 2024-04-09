# routes.py
import base64
import io
import uuid

import cv2
import numpy as np
from flask import jsonify, request
from PIL import Image

from app import app
from app.dummy_data.responses import get_dummy_bounding_boxes

# Assuming segment_image is your actual segmentation function you'd like to use
from app.utils.image_processing import segment_image
from app.utils.path_construction import (
    construct_path,
    create_graph_from_image,
    tsp_heuristic,
)

# Global variable to store the graph
global_graph_store = {}
image_shape = [0, 0]
white_threshold = 245  # Threshold for walkable region


@app.route("/status", methods=["GET"])
def server_status():
    return jsonify({"message": "Server running"}), 200


@app.route("/segment", methods=["POST"])
def segment_and_graph_route():
    data = request.get_json()

    unique_id = str(uuid.uuid4())

    if not data or "image" not in data:
        return jsonify({"error": "Missing image in request"}), 400

    try:
        # Perform image segmentation
        # This is where you would typically call your segmentation function.
        # For demonstration, let's assume segment_image returns bounding_boxes and modifies the image for graph creation.
        # bounding_boxes = segment_image(data["image_path"])
        bounding_boxes = (
            get_dummy_bounding_boxes()
        )  # Replace with actual call to segment_image

        if data["image"].startswith("data:image"):
            base64_str = data["image"].split(",", 1)[1]
        else:
            base64_str = data["image"]

        # Decode the base64 string
        image_bytes = base64.b64decode(base64_str)

        # Convert bytes to a NumPy array
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)

        # Read the image from the NumPy array
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        image_shape[0] = image.shape[0]
        image_shape[1] = image.shape[1]

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Identify traversable pixels
        traversable_pixels = gray_image > white_threshold

        # Generate the graph from the segmented (and possibly modified) image
        G = create_graph_from_image(traversable_pixels)

        # Store G in a global variable using a unique identifier, for example, the modified image path
        global_graph_store[unique_id] = G

        return jsonify(
            {
                "message": "Segmentation and graph creation successful",
                "image_path": unique_id,
                "bounding_boxes": bounding_boxes,  # Assuming bounding_boxes is the result you want to return
            }
        )
    except Exception as e:
        return jsonify({"error": "Failed to process image", "details": str(e)}), 500


@app.route("/construct-path", methods=["POST"])
def find_path_route():
    data = request.get_json()
    if (
        not data
        or "start_point" not in data
        or "grocery_cart" not in data
        or "image_path" not in data
    ):
        return jsonify({"error": "Missing data for path finding"}), 400

    try:
        # canvas_width = data.get("canvas_width")
        # canvas_height = data.get("canvas_height")

        # height, width = image_shape

        # Scale factor between the original image and the canvas
        # x_scale = width / canvas_width
        # y_scale = height / canvas_height

        # print(x_scale)
        # print(y_scale)
        # print(data["grocery_cart"])
        # grocery_cart_locs = [
        #     [item["y"] * y_scale, item["x"] * x_scale] for item in data["grocery_cart"]
        # ]
        # start_point = [
        #     data["start_point"][0] * x_scale,
        #     data["start_point"][1] * y_scale,
        # ]

        grocery_cart_locs = [(item["y"], item["x"]) for item in data["grocery_cart"]]

        print(grocery_cart_locs)

        # Scale the start_point and grocery_cart_locs
        start_point = (
            data["start_point"][0],
            data["start_point"][1],
        )

        image_path = data["image_path"]
        if image_path not in global_graph_store:
            return jsonify({"error": "Graph not found for the given image path"}), 404

        G = global_graph_store[image_path]

        # grocery_cart_locs = [
        #     [(coord[0] * x_scale, coord[1] * y_scale) for coord in item["coords"]]
        #     for item in data["grocery_cart"]
        # ]

        # item_position_list = {
        #     item: (coord[1], coord[0])
        #     for item, coord in data.get("item_position_list", {}).items()
        # }
        # grocery_cart_locs = [
        #     (item_position_list[item][1], item_position_list[item][0])
        #     for item in grocery_cart
        #     if item in item_position_list
        # ]

        if not grocery_cart_locs:
            return (
                jsonify(
                    {
                        "error": "Invalid or missing locations for items in the grocery cart"
                    }
                ),
                400,
            )

        visit_order, total_distance = tsp_heuristic(
            G, (start_point[1], start_point[0]), grocery_cart_locs
        )
        optimal_pick_up_path = construct_path(G, visit_order)

        optimal_pick_up_path_reversed = [
            (node[1], node[0]) for node in optimal_pick_up_path
        ]
        return jsonify(
            {
                "visit_order": visit_order,
                "total_distance": total_distance,
                "optimal_pick_up_path": optimal_pick_up_path_reversed,
            }
        )
    except Exception as e:
        return jsonify({"error": "Failed to find path", "details": str(e)}), 500
