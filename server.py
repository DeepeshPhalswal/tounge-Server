import os
from datetime import datetime
from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import base64
from dotenv import load_dotenv
from predict import predict_image

# ------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# ------------------------------------------
load_dotenv()

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
PROCESSED_DIR = os.getenv("PROCESSED_DIR", "processed")
FLASK_PORT = int(os.getenv("FLASK_PORT", 8000))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "False").lower() == "true"

EDGE_THRESHOLD1 = int(os.getenv("EDGE_THRESHOLD1", 100))
EDGE_THRESHOLD2 = int(os.getenv("EDGE_THRESHOLD2", 200))
AI_OVERLAY_WEIGHT = float(os.getenv("AI_OVERLAY_WEIGHT", 0.3))

# Ensure folders exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

app = Flask(__name__)

# ------------------------------------------
# AI IMAGE PROCESSING FUNCTION
# ------------------------------------------
def process_image_ai(image):
    """Run AI + OpenCV processing."""
    try:
        # AI Prediction
        result = predict_image(image)
    except Exception as e:
        result = f"AI Prediction Error: {str(e)}"

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, EDGE_THRESHOLD1, EDGE_THRESHOLD2)

    # Brightness & contrast
    mean_brightness = np.mean(gray)
    contrast = np.std(gray)

    # Overlay edges in red
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(image, 1 - AI_OVERLAY_WEIGHT, edges_colored, AI_OVERLAY_WEIGHT, 0)

    ai_text = (
        f"🧠 AI Analysis:\n"
        f"- Result: {result}\n"
        f"- Mean Brightness: {mean_brightness:.2f}\n"
        f"- Contrast: {contrast:.2f}\n"
        f"- Edge Thresholds: ({EDGE_THRESHOLD1}, {EDGE_THRESHOLD2})"
    )

    return overlay, ai_text

# ------------------------------------------
# API ROUTES
# ------------------------------------------
@app.route("/")
def home():
    return "<h2>🧠 AI Image Processing Server is Running</h2>"

@app.route("/process_image", methods=["POST"])
def process_image():
    """Handles image upload and AI processing."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
    upload_path = os.path.join(UPLOAD_DIR, filename)
    file.save(upload_path)

    img = cv2.imread(upload_path)
    if img is None:
        return jsonify({"error": "Invalid image format"}), 400

    # Process image
    processed_img, result_text = process_image_ai(img)

    # Save processed image
    processed_name = "processed_" + filename
    processed_path = os.path.join(PROCESSED_DIR, processed_name)
    cv2.imwrite(processed_path, processed_img)

    # Convert processed image to base64
    _, buffer = cv2.imencode(".jpg", processed_img)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return jsonify({
        "result": result_text,
        "processed_image_base64": img_base64,
        "processed_image_url": f"/get_processed/{processed_name}"
    })

@app.route("/get_processed/<filename>")
def get_processed(filename):
    """Serve processed image."""
    path = os.path.join(PROCESSED_DIR, filename)
    if not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404
    return send_file(path, mimetype="image/jpeg")

# ------------------------------------------
# MAIN ENTRY POINT
# ------------------------------------------
if __name__ == "__main__":
    print("🚀 Starting AI Image Processing Server...")
    print(f"📁 Upload directory: {UPLOAD_DIR}")
    print(f"📁 Processed directory: {PROCESSED_DIR}")
    print(f"🌐 Running on port {FLASK_PORT} (Debug={FLASK_DEBUG})")
    app.run(host="0.0.0.0", port=FLASK_PORT, debug=FLASK_DEBUG)
