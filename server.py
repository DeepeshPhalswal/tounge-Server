from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import io
from PIL import Image
import base64
import os
from datetime import datetime

app = Flask(__name__)

# Directory to save uploaded and processed images
UPLOAD_DIR = "uploads"
PROCESSED_DIR = "processed"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ------------------------------------------
# AI / IMAGE PROCESSING LOGIC
# ------------------------------------------
def process_image_ai(image):
    """
    Example AI image processing function.
    Replace this with your own AI model inference.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Example 1: Edge detection (for visualization)
    edges = cv2.Canny(gray, 100, 200)

    # Example 2: Compute brightness and contrast info
    mean_brightness = np.mean(gray)
    contrast = np.std(gray)

    # Create color overlay for edges
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    processed = cv2.addWeighted(image, 0.7, edges_colored, 0.3, 0)

    # Generate result text (replace this with model output if needed)
    ai_text = (
        f"🧠 AI Analysis:\n"
        f"- Mean brightness: {mean_brightness:.2f}\n"
        f"- Contrast: {contrast:.2f}\n"
        f"- Edge detection applied successfully ✅"
    )

    return processed, ai_text

# ------------------------------------------
# API ROUTES
# ------------------------------------------
@app.route("/")
def home():
    return "<h2>🧠 AI Image Processing Server is Running</h2>"

@app.route("/process_image", methods=["POST"])
def process_image():
    """
    Receives an image from the Raspberry Pi app, processes it,
    and returns AI text + processed image.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
    filepath = os.path.join(UPLOAD_DIR, filename)
    file.save(filepath)

    # Read image
    img = cv2.imread(filepath)
    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    # AI Processing
    processed_img, result_text = process_image_ai(img)

    # Save processed image
    processed_path = os.path.join(PROCESSED_DIR, "processed_" + filename)
    cv2.imwrite(processed_path, processed_img)

    # Convert processed image to base64 (to send in response)
    _, buffer = cv2.imencode(".jpg", processed_img)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    # Return JSON response
    response = {
        "result": result_text,
        "processed_image_base64": img_base64,
        "processed_image_url": f"/get_processed/{'processed_' + filename}"
    }
    return jsonify(response)

@app.route("/get_processed/<filename>")
def get_processed(filename):
    """
    Serve processed image for viewing via URL.
    """
    path = os.path.join(PROCESSED_DIR, filename)
    if not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404
    return send_file(path, mimetype="image/jpeg")

# ------------------------------------------
# RUN SERVER
# ------------------------------------------
if __name__ == "__main__":
    print("🚀 Starting AI Image Processing Server on port 8000...")
    app.run(host="0.0.0.0", port=8000)
