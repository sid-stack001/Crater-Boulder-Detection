# Crater & Boulder Detection

A polished AI project for detecting craters and boulders in lunar or planetary imagery using YOLO-based object detection. The repository is structured to make it easy to run locally, showcase in interviews, and present professionally on GitHub or a resume.

## Overview

This project combines computer vision and deep learning to identify hazardous geological features in high-resolution images of planetary surfaces. It is designed for lunar landing site analysis, terrain assessment, and autonomous exploration support.

## Why this project stands out

- End-to-end deep-learning workflow for crater and boulder detection
- Tile-based processing for larger images
- Exportable detection outputs in XML and label summaries
- Interactive web interface for image upload and inference
- Clean project layout suitable for portfolio and resume presentation

## Tech stack

- Python
- OpenCV
- NumPy
- Pillow
- Streamlit
- Ultralytics YOLO

## Repository structure

```text
.
├── crater_detection/        # Core app package
│   ├── __init__.py
│   ├── app.py               # Streamlit application entrypoint
│   ├── config.py            # Centralized model configuration
│   └── inference.py         # Detection and export utilities
├── models/                  # Model weights location
│   └── README.md
├── .env.example             # Example environment configuration
├── .gitignore               # Clean repo hygiene
├── app.py                   # Root app launch wrapper
├── app_streamlit.py         # Compatibility wrapper
├── requirements.txt         # Dependency list
├── LICENSE
├── README.md
├── resources/               # Research papers and supporting material
├── yolo_model/              # Local YOLO training outputs and artifacts
├── *.ipynb                  # Legacy exploratory notebooks retained for reference
└── ...
```

## Setup

1. Clone the repository.
2. Create a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Add the trained YOLO weights to `models/best.pt`.

If you want to use a different location, set:

```bash
export CRATER_MODEL_PATH="/path/to/your/model/best.pt"
```

## Run the app

```bash
streamlit run src/streamlit_app.py
```

or:

```bash
python app.py
```

## Model requirements

The trained `.pt` file is intentionally not committed to GitHub because of its size. Place it in the `models/` directory or point `CRATER_MODEL_PATH` to your local copy.

## Example outputs

The repository includes sample images and generated results from exploratory runs for visualization and validation.

## Project impact

This project is relevant to:

- Planetary science and lunar geology
- Hazard mapping for landing zones
- Computer vision in remote sensing
- AI-driven autonomous exploration workflows

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
