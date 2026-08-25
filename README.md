# Crater & Boulder Detection

Professional implementation for automated detection of craters and boulders in high-resolution planetary imagery using Ultralytics YOLO and standard image-processing libraries.

---

## Overview

This repository provides a compact, reproducible pipeline for processing large planetary images, running tile-based object detection, producing annotated visualizations, and exporting structured detection outputs. The project is suitable for demonstrations, portfolio inclusion, and prototype evaluation for tasks such as lunar landing-site assessment and hazard mapping.

---

## Repository contents

- [src/](F:/project/crater/Crater-Boulder-Detection.worktrees/repo-cleanup-and-professionalization/src) — Streamlit and Flask entry scripts.
- [models/](F:/project/crater/Crater-Boulder-Detection.worktrees/repo-cleanup-and-professionalization/models) — Trained model weights and related artifacts (not committed).
- [notebooks/](F:/project/crater/Crater-Boulder-Detection.worktrees/repo-cleanup-and-professionalization/notebooks) — Exploratory Jupyter notebooks used during development.
- [assets/images/](F:/project/crater/Crater-Boulder-Detection.worktrees/repo-cleanup-and-professionalization/assets/images) — Example input images and generated visual outputs.
- [assets/results/](F:/project/crater/Crater-Boulder-Detection.worktrees/repo-cleanup-and-professionalization/assets/results) — Inference outputs produced by the application.
- [assets/videos/](F:/project/crater/Crater-Boulder-Detection.worktrees/repo-cleanup-and-professionalization/assets/videos) — Demo recordings.
- [web/](F:/project/crater/Crater-Boulder-Detection.worktrees/repo-cleanup-and-professionalization/web) — Static demo pages.
- [resources/](F:/project/crater/Crater-Boulder-Detection.worktrees/repo-cleanup-and-professionalization/resources) — Research papers and references.
- [LICENSE](F:/project/crater/Crater-Boulder-Detection.worktrees/repo-cleanup-and-professionalization/LICENSE)

---

## Quick start

1. Create and activate a Python virtual environment (recommended):

   Windows PowerShell

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   macOS / Linux

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install the required packages (if a requirements.txt is not present, install these core packages):

```bash
pip install opencv-python pillow numpy streamlit ultralytics
```

3. Place your trained YOLO weights at `models/best.pt` or set the environment variable `CRATER_MODEL_PATH` to the absolute path of the .pt file.

Example (Windows PowerShell):

```powershell
$env:CRATER_MODEL_PATH = 'C:\path\to\best.pt'
```

---

## Usage

Interactive (Streamlit)

```bash
streamlit run src/streamlit_app.py
```

The Streamlit application provides an interface to upload an image, invoke detection, preview annotated results, and download PNG/XML outputs.

Programmatic (Flask)

```bash
python src/flask_app.py
```

Send a POST request (form field `file`) to `/upload`. The endpoint returns a PNG image with visualized detections and saves a copy to `assets/results/`.

---

## Examples

Sample input:

![Sample input](F:/project/crater/Crater-Boulder-Detection.worktrees/repo-cleanup-and-professionalization/assets/images/sample_input.jpg)

Segmentation example:

![Segmentation example](F:/project/crater/Crater-Boulder-Detection.worktrees/repo-cleanup-and-professionalization/assets/images/segmentation_example.png)

Predicted output (high resolution):

![Predicted high resolution](F:/project/crater/Crater-Boulder-Detection.worktrees/repo-cleanup-and-professionalization/assets/images/predicted_highres.png)

User interface preview:

![UI screenshot 1](F:/project/crater/Crater-Boulder-Detection.worktrees/repo-cleanup-and-professionalization/assets/images/ui_screenshot_1.png)

![UI screenshot 2](F:/project/crater/Crater-Boulder-Detection.worktrees/repo-cleanup-and-professionalization/assets/images/ui_screenshot_2.png)

---

## Processing pipeline (overview)

The following diagram summarizes the main data flow used by the applications and notebooks:

```
+--------------------+
| 1) Input image     |
+--------------------+
          |
          v
+--------------------+
| 2) Tile splitter   |
|    (fixed-size     |
|     tiles + overlap)
+--------------------+
          |
          v
+--------------------+
| 3) YOLO inference  |
|    (per-tile model |
|     prediction)    |
+--------------------+
          |
          v
+--------------------+    +--------------------+
| 4) Postprocessing  | -> | 5) Outputs          |
|    (merge, NMS,    |    | - annotated PNG     |
|     format export) |    | - XML annotations   |
+--------------------+    | - summary text file |
                          +--------------------+
```

This diagram is intended as a compact reference for the repository structure and runtime flow.

---

## License

This project is distributed under the MIT License. See [LICENSE](F:/project/crater/Crater-Boulder-Detection.worktrees/repo-cleanup-and-professionalization/LICENSE) for details.

---

