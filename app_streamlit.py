import base64
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import io
import cv2
from ultralytics import YOLO
import xml.etree.ElementTree as ET

# Function to convert a bounding box to a circle
def box_to_circle(x1, y1, x2, y2):
    center = (int(x1 + (x2 - x1) / 2), int(y1 + (y2 - y1) / 2))
    radius = int(min(x2 - x1, y2 - y1) / 2)
    return center, radius

# Function to run YOLO on image tiles
def run_yolo_on_tiles(image, model, tile_size=640, overlap=0.2):
    height, width = image.shape[:2]
    stride = int(tile_size * (1 - overlap))
    detections = []

    for y in range(0, height, stride):
        for x in range(0, width, stride):
            x_end = min(x + tile_size, width)
            y_end = min(y + tile_size, height)
            tile = image[y:y_end, x:x_end]
            results = model(tile)
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].clone()
                cls = int(box.cls[0])  # Get the class ID
                x1 += x
                y1 += y
                x2 += x
                y2 += y
                detections.append((x1, y1, x2, y2, cls))

    return detections

# Function to create XML from detections
def create_xml(detections, class_names):
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
        diameter_meters = diameter_pixels * 0.32  # Convert to meters using resolution
        ET.SubElement(obj, "DiameterMeters").text = f'{diameter_meters:.2f}'

    return ET.tostring(root, encoding='utf8', method='xml')

# Load the YOLO model
model = YOLO('G:/yolo_model/runs/detect/train/weights/best.pt')

# Custom CSS for a more colorful and appealing design with a background image
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("image.jpg")

st.markdown("""
    <style>
    .title-container {
        display: flex;
        justify-content: flex-start;
        align-items: left;
    }
    .title-container h1 {
        font-size: 5em;
        font-weight: bold;
        margin: 0;
        color:#003B5C ; /* Change this color as needed */
        font-family: 'Courier New', monospace;
    }
    </style>
    <div class="title-container">
        <h1>AKASHDRUSHTI</h1>
    </div>
""", unsafe_allow_html=True)

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://images.unsplash.com/photo-1451187580459-43490279c0fa?fm=jpg&w=3000&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8ZWFydGglMjBhdCUyMG5pZ2h0fGVufDB8fDB8fHww");
background-size: 100%;
background-repeat: no-repeat;
# background-attachment: local;
}}

[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:image/png;base64,{img}");
background-position: center; 
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0,255,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# Custom CSS for changing font colors
# st.title('      GAGANDRISHTI')
st.title("Crater and Boulder Detection App")

st.markdown("## 🌌 Upload and Process Your High-Resolution Space Images")
st.write("Upload your high-resolution images and use our tools to detect craters and boulders, apply various transformations, and download the processed images along with the detected features.")

# Upload image
uploaded_file = st.file_uploader("Upload a high-resolution image", type=["png", "jpg", "jpeg", "tiff"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert the uploaded PIL image to OpenCV format
    image_cv = np.array(image.convert('RGB'))
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    
    # Image processing options
    st.markdown("### 🛠️ Image Processing Options")
    col1, col2 = st.columns(2)
    with col1:
        from PIL import Image
        import numpy as np
        import streamlit as st
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        # Assuming 'image' is already defined as a PIL Image object
        if st.button("Convert to Rainbow"):
            # Convert the image to grayscale
            image_gray = ImageOps.grayscale(image)
            
            # Convert the grayscale image to a NumPy array
            img_array_gray = np.array(image_gray)
            
            # Normalize the grayscale image to [0, 1] range
            img_array_normalized = img_array_gray / 255.0
            
            # Apply the rainbow colormap
            cmap = plt.get_cmap('rainbow')
            img_array_colored = cmap(img_array_normalized)  # Apply colormap
            
            # Convert the colormapped array back to an image
            img_array_colored = (img_array_colored[:, :, :3] * 255).astype(np.uint8)  # Drop alpha channel and convert to 8-bit
            rainbow_image = Image.fromarray(img_array_colored)
            
            # Show the rainbow-toned image
            st.image(rainbow_image, caption='Rainbow-Toned Image', use_column_width=True)



        if st.button("Crop Image"):
            width, height = image.size
            left = width / 4
            top = height / 4
            right = 3 * width / 4
            bottom = 3 * height / 4
            image = image.crop((left, top, right, bottom))
            st.image(image, caption='Cropped Image', use_column_width=True)
    
    with col2:
        st.write("Choose options for mapping, visual color changes, and exploring the image (features to be added).")

    # Detection
    st.markdown("### 🔍 Detection")
    if st.button("Detect Craters and Boulders"):
        # Run YOLO on image tiles
        tile_size = 640
        overlap = 0.2
        detections = run_yolo_on_tiles(image_cv, model, tile_size, overlap)
        
        # Define a list of class names (update this list with the actual class names used by your model)
        class_names = ["boulder", "crater"]

        # Variables to count craters and boulders
        num_craters = 0
        num_boulders = 0

        # Draw circles and labels on the image
        for x1, y1, x2, y2, cls in detections:
            center, radius = box_to_circle(x1, y1, x2, y2)
            cv2.circle(image_cv, center, radius, (255, 0, 0), 2)  # Draw blue circle

            # Calculate diameter in pixels and convert to meters
            diameter_pixels = radius * 2
            diameter_meters = diameter_pixels * 0.32  # Convert to meters using resolution

            # Draw label background
            label = f'{class_names[cls]}: {diameter_meters:.2f} m'
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(image_cv, (center[0] - w // 2, center[1] - radius - 20), 
                          (center[0] + w // 2, center[1] - radius - 20 + h), (255, 0, 0), -1)

            # Draw label text
            cv2.putText(image_cv, label, (center[0] - w // 2, center[1] - radius - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Count craters and boulders
            if class_names[cls] == 'crater':
                num_craters += 1
            elif class_names[cls] == 'boulder':
                num_boulders += 1

        # Display the number of craters and boulders
        st.write(f"Number of Craters: {num_craters}")
        st.write(f"Number of Boulders: {num_boulders}")

        # Convert the processed image back to PIL format for display
        processed_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        processed_image = Image.fromarray(processed_image)
        st.image(processed_image, caption='Processed Image with Detection', use_column_width=True)

        # Convert the processed image to a downloadable format
        buf = io.BytesIO()
        processed_image.save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.download_button(label="Download Processed Image", data=byte_im, file_name="processed_image.png", mime="image/png")

        # Create XML and provide download option
        xml_data = create_xml(detections, class_names)
        st.download_button(label="Download Detection Data as XML", data=xml_data, file_name="detection_data.xml", mime="application/xml")

        # Download labels
        labels = f"Craters: {num_craters}\nBoulders: {num_boulders}\n"
        st.download_button(label="Download Labels", data=labels, file_name="labels.txt", mime="text/plain")

    st.markdown("### 🚀 Future Enhancements")
    st.write("Custom download options and more features coming soon!")

else:
    st.markdown("### 📤 Please upload an image to proceed.")

# Footer
st.markdown("""
    <hr>
    <center>
    Made with by TEAM AKASHDRUSHTI
    </center>
    """, unsafe_allow_html=True)
