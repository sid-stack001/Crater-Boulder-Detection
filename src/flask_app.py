from flask import Flask, request, jsonify, send_file, render_template, send_from_directory
import cv2
import numpy as np
import os
from pathlib import Path

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

app = Flask(__name__, static_folder='../web', template_folder='../web')


def box_to_circle(x1, y1, x2, y2):
    center = (int(x1 + (x2 - x1) / 2), int(y1 + (y2 - y1) / 2))
    radius = int(min(x2 - x1, y2 - y1) / 2)
    return center, radius


def find_model_path() -> Path | None:
    env = os.getenv('CRATER_MODEL_PATH')
    if env:
        p = Path(env)
        if p.exists():
            return p
    candidates = [
        Path('models') / 'best.pt',
        Path('models') / 'runs' / 'detect' / 'train' / 'weights' / 'best.pt',
        Path('models') / 'yolov8n.pt',
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


MODEL_PATH = find_model_path()
MODEL = None
if YOLO is not None and MODEL_PATH is not None:
    try:
        MODEL = YOLO(str(MODEL_PATH))
    except Exception:
        MODEL = None


@app.route('/')
def index():
    # If web/index.html exists, serve it; otherwise a small message
    idx = Path(app.template_folder) / 'index.html'
    if idx.exists():
        return render_template('index.html')
    return 'Crater Detection Flask API. Use /upload to POST an image.'


@app.route('/upload', methods=['POST'])
def upload_file():
    if MODEL is None:
        return jsonify({"error": "Model not available. Place the weights in the models/ folder or set CRATER_MODEL_PATH."}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    data = file.read()
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Failed to decode image"}), 400

    results = MODEL(img)
    detections = results[0].boxes

    for box in detections:
        coords = box.xyxy[0].tolist()
        x1, y1, x2, y2 = coords
        center, radius = box_to_circle(x1, y1, x2, y2)
        cv2.circle(img, center, radius, (255, 0, 0), 2)
        diameter_pixels = radius * 2
        diameter_meters = diameter_pixels * 0.32
        label = f'{diameter_meters:.2f} m'
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (center[0] - w // 2, center[1] - radius - 20), (center[0] + w // 2, center[1] - radius - 20 + h), (255, 0, 0), -1)
        cv2.putText(img, label, (center[0] - w // 2, center[1] - radius - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    out_dir = Path('assets') / 'results'
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / 'predicted_circle.png'
    cv2.imwrite(str(output_path), img)

    return send_file(str(output_path), mimetype='image/png')


@app.route('/<path:path>')
def static_proxy(path):
    return send_from_directory('.', path)


if __name__ == '__main__':
    app.run(debug=True)
