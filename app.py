from flask import Flask, request, jsonify, send_file, render_template, send_from_directory
import cv2
import numpy as np
from ultralytics import YOLO
import os

app = Flask(__name__)

# Function to convert a bounding box to a circle
def box_to_circle(x1, y1, x2, y2):
    center = (int(x1 + (x2 - x1) / 2), int(y1 + (y2 - y1) / 2))
    radius = int(min(x2 - x1, y2 - y1) / 2)
    return center, radius

# Load the YOLO model
model = YOLO('G:/yolo_model/runs/detect/train/weights/best.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    if file:
        # Read image file
        img = np.fromstring(file.read(), np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)

        # Perform prediction
        results = model(img)

        # Extract bounding boxes and labels
        detections = results[0].boxes

        # Draw circles and labels on the image
        for box in detections:
            x1, y1, x2, y2 = box.xyxy[0]
            center, radius = box_to_circle(x1, y1, x2, y2)
            cv2.circle(img, center, radius, (255, 0, 0), 2)  # Draw blue circle

            # Calculate diameter in pixels and convert to meters
            diameter_pixels = radius * 2
            diameter_meters = diameter_pixels * 0.32  # Convert to meters using resolution

            # Draw label background
            label = f'{diameter_meters:.2f} m'
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (center[0] - w // 2, center[1] - radius - 20), 
                          (center[0] + w // 2, center[1] - radius - 20 + h), (255, 0, 0), -1)

            # Draw label text
            cv2.putText(img, label, (center[0] - w // 2, center[1] - radius - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # Save the modified image to a file
        output_path = "predicted_circle.png"
        cv2.imwrite(output_path, img)

        return send_file(output_path, mimetype='image/png')

@app.route('/<path:path>')
def static_proxy(path):
    return send_from_directory('.', path)

if __name__ == '__main__':
    app.run(debug=True)
