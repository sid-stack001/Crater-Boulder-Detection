# Crater-Boulder-Detection
 
## 📌 Overview  
This is an AI-powered **crater and boulder detection system** designed for analyzing high-resolution **lunar surface images**. By leveraging deep learning, the software identifies geological features critical for **lunar exploration, landing site selection, and hazard assessment**.  

## 🔍 Problem Statement  
Lunar missions require **precise hazard detection** for safe landings and surface exploration. Traditional manual mapping is **time-consuming and inconsistent**. This system automates detection using advanced **computer vision** and **machine learning models**, making the process **efficient, scalable, and accurate**.  

## 🚀 Features  
- 🛰 **Crater & Boulder Detection** – Identifies key surface features in lunar images  
- 🔄 **Tile-Based Image Processing** – Handles large images by processing smaller tiles  
- 📊 **Size & Depth Estimation** – Computes diameters and dimensions using **DEM scaling**  
- 📍 **Geospatial Mapping** – Provides coordinates for detected features  
- 🎨 **Custom Image Transformations** – Includes rainbow-toned visualization, cropping, and more  
- 📂 **Automated Data Export** – Saves detections as **XML, text, and processed images**  

## 🛠 Tech Stack  
- **Python** 🐍 – Core programming language  
- **YOLO (Ultralytics)** 🛰 – Deep learning model for object detection  
- **OpenCV** 👁 – Image processing and feature extraction  
- **Streamlit** 🌐 – Web-based interactive UI  
- **PIL & NumPy** 🖼 – Image transformations and processing  
- **CUDA** ⚡ – GPU acceleration for real-time detection  

## ⚙️ How It Works  
1️⃣ **Image Upload** – Users upload high-resolution lunar images  
2️⃣ **Image Preprocessing** – Enhances images and converts them into analyzable formats  
3️⃣ **YOLO Detection** – The model detects craters and boulders from processed tiles  
4️⃣ **Feature Annotation** – Outputs detected objects with bounding circles & labels  
5️⃣ **Size & Depth Estimation** – Computes diameters using pixel-to-meter scaling  
6️⃣ **Result Export** – Processed images, XML detection data, and reports are available for download  


Note: for running the code, the trained yolo .pt file is required, which is not uploaded, due to size constraints. It is available [here](https://drive.google.com/file/d/1LGhTr1WuqYC2eNBksiMvA9T_ptEQnFKf/view?usp=drive_link).



<img width="1280" alt="Screenshot 2024-07-25 160643" src="https://github.com/user-attachments/assets/64da3b6a-6a93-4a6a-9456-e34c82466999" />


<img width="718" alt="Screenshot 2024-07-25 151332" src="https://github.com/user-attachments/assets/a3432359-208e-40eb-9f8e-ab1c51df493b" />




