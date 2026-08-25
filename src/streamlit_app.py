import base64
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import io
import cv2
import os
from pathlib import Path

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


def box_to_circle(x1, y1, x2, y2):
    center = (int(x1 + (x2 - x1) / 2), int(y1 + (y2 - y1) / 2))
    radius = int(min(x2 - x1, y2 - y1) / 2)
    return center, radius


def run_yolo_on_tiles(image, model, tile_size=640, overlap=0.2):
    height, width = image.shape[:2]
    stride = max(1, int(tile_size * (1 - overlap)))
    detections = []

    for y in range(0, height, stride):
        for x in range(0, width, stride):
            x_end = min(x + tile_size, width)
            y_end = min(y + tile_size, height)
            tile = image[y:y_end, x:x_end]
            results = model(tile)
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls = int(box.cls[0]) if hasattr(box, 'cls') else 0
                detections.append((x1 + x, y1 + y, x2 + x, y2 + y, cls))

    return detections


def create_xml(detections, class_names):
    import xml.etree.ElementTree as ET
    root = ET.Element("Detections")
    for x1, y1, x2, y2, cls in detections:
        obj = ET.SubElement(root, "Object")
        ET.SubElement(obj, "Class").text = class_names[cls]
        ET.SubElement(obj, "X1").text = str(x1)
        ET.SubElement(obj, "Y1").text = str(y1)
        ET.SubElement(obj, "X2").text = str(x2)
        ET.SubElement(obj, "Y2").text = str(y2)
        center, radius = box_to_circle(x1, y1, x2, y2)
        diameter_pixels = radius * 2
        diameter_meters = diameter_pixels * 0.32
        ET.SubElement(obj, "DiameterMeters").text = f"{diameter_meters:.2f}"
    return ET.tostring(root, encoding='utf8', method='xml')


def find_model_path() -> Path | None:
    env = os.getenv('CRATER_MODEL_PATH')
    if env:
        p = Path(env)
        if p.exists():
            return p
    candidates = [Path('models') / 'best.pt', Path('models') / 'runs' / 'detect' / 'train' / 'weights' / 'best.pt']
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


st.set_page_config(page_title="Crater & Boulder Detection", page_icon="🌙", layout="wide")

st.title("Crater and Boulder Detection")
st.caption("Lunar surface analysis for geologic feature identification and hazard assessment.")

uploaded_file = st.file_uploader("Upload a lunar image", type=["png", "jpg", "jpeg", "tiff"])
if uploaded_file is None:
    st.info("Please upload an image to begin analysis.")
else:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    image_cv = np.array(image)
    image_bgr = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    if st.button("Detect craters and boulders"):
        if MODEL is None:
            st.error("Model weights not found. Place the trained weights in the models/ folder or set CRATER_MODEL_PATH.")
        else:
            detections = run_yolo_on_tiles(image_bgr, MODEL, tile_size=640, overlap=0.2)
            class_names = ["boulder", "crater"]
            num_craters = sum(1 for _, _, _, _, cls in detections if class_names[cls] == "crater")
            num_boulders = sum(1 for _, _, _, _, cls in detections if class_names[cls] == "boulder")

            annotated = image_bgr.copy()
            for x1, y1, x2, y2, cls in detections:
                center, radius = box_to_circle(x1, y1, x2, y2)
                cv2.circle(annotated, center, radius, (255, 0, 0), 2)
                label = f"{class_names[cls]}: {radius * 2 * 0.32:.2f} m"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(annotated, (center[0] - w // 2, center[1] - radius - 20), (center[0] + w // 2, center[1] - radius - 20 + h), (255, 0, 0), -1)
                cv2.putText(annotated, label, (center[0] - w // 2, center[1] - radius - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            processed_image = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            processed_pil = Image.fromarray(processed_image)
            st.image(processed_pil, caption='Processed image with detections', use_column_width=True)

            buf = io.BytesIO()
            processed_pil.save(buf, format='PNG')
            st.download_button(label='Download processed image', data=buf.getvalue(), file_name='processed_image.png', mime='image/png')

            xml_data = create_xml(detections, class_names)
            st.download_button(label='Download detection XML', data=xml_data, file_name='detection_data.xml', mime='application/xml')

            labels = f"Craters: {num_craters}\nBoulders: {num_boulders}\n"
            st.download_button(label='Download summary labels', data=labels, file_name='summary_labels.txt', mime='text/plain')
