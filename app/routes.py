from flask import jsonify, request

from app import app
from app.dummy_data.responses import get_dummy_bounding_boxes

# from app.utils.image_processing import segment_and_text_detection
from app.utils.image_processing import segment_image


@app.route("/status", methods=["GET"])
def server_status():
    return jsonify({"message": "Server running"}), 200


@app.route("/segment", methods=["POST"])
def segment_route():
    data = request.get_json()

    if not data or "image" not in data:
        return jsonify({"error": "Missing image data"}), 400

    try:
        # segments = segment_and_text_detection(data["image"])

        # print(segments)
        # return jsonify(segments)
        bounding_boxes = get_dummy_bounding_boxes()
        return jsonify(bounding_boxes)
    except Exception as e:
        print(e)
        return jsonify({"error": "Segmentation failed", "details": str(e)}), 500
