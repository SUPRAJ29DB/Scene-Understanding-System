# Project README
# AI-Based Road Scene Understanding and Risk Analysis

An AI-powered road-scene analysis prototype combining **YOLO11 object detection**, **2D scene-graph construction**, **rule-based risk analysis**, and a **BLIP vision-language model (VLM)**. It generates annotated images and a structured text report, with a Streamlit dashboard for interactive use.

> **Status:** Working prototype. This is a research/educational system, not a certified road-safety or collision-prediction product.
> 

## Table of Contents

- Overview
- Features
- Architecture
- Technology Stack
- Dataset and Classes
- Model Results
- Project Structure
- Installation
- Configuration
- Run the Project
- Outputs
- Risk Score Interpretation
- Limitations
- Troubleshooting
- Future Improvements
- License

## Overview

The system analyzes a road image through the following stages:

1. YOLO11 detects supported road-scene object classes.
2. A scene graph represents detections as nodes and 2D spatial relations as edges.
3. A rule-based risk analyzer evaluates selected proximity, overlap, and scene-density indicators.
4. BLIP produces a general visual caption.
5. A report generator combines the caption, detected objects, scene relationships, and risk events.
6. A Streamlit dashboard displays the results and provides report/image downloads.

## Features

- YOLO11 road-object detection.
- 2D scene graph with `left_of`, `above`, `below`, `near`, and `overlapping` relationships.
- Heuristic risk levels and scores.
- BLIP image captioning using Hugging Face Transformers.
- Annotated images with detections, selected relationships, and risk summary.
- Text report generation.
- Streamlit interface with configurable confidence threshold and maximum detections.
- CUDA support when a compatible GPU/PyTorch installation is available; VLM module supports CPU fallback.

## Architecture

```
Input Road Image
   |-------------------------|
   v                         v
YOLO11 Detection           BLIP VLM
   |                         |
Detected Objects         Visual Caption
   |
Scene Graph
   |
Spatial Relationships
   |
Rule-Based Risk Analyzer
   |-------------------------|
               v
        Report Generator
               |
       +-------+--------+
       v                v
  Text Report     Annotated Image
       |                |
       +-------+--------+
               v
       Streamlit Dashboard
```

## Technology Stack

- Python 3.10
- Ultralytics YOLO11
- PyTorch
- Hugging Face Transformers and BLIP
- OpenCV and Pillow
- Streamlit
- BDD100K images and labels
- Anaconda/virtual environment and PyCharm (development)

## Dataset and Classes

The detector was trained on a prepared subset converted from BDD100K. The current prepared subset contained approximately **1,153 training images** and **2,000 validation images**. These are the local subset counts, not the full BDD100K dataset.

| ID | Class |
| --- | --- |
| 0 | pedestrian |
| 1 | rider |
| 2 | car |
| 3 | truck |
| 4 | bus |
| 5 | train |
| 6 | motorcycle |
| 7 | bicycle |
| 8 | traffic light |
| 9 | traffic sign |

Dataset preparation script: `scripts/prepare_bdd100k.py`

YOLO configuration: `dataset/data.yaml`

Example `dataset/data.yaml`:

```yaml
path: dataset
train: images/train
val: images/val

names:
0: pedestrian
1: rider
2: car
3: truck
4: bus
5: train
6: motorcycle
7: bicycle
8: traffic light
9: traffic sign
```

Download BDD100K from its official distribution and follow its license and access terms. Do not commit the complete dataset to GitHub unless its terms and repository limits allow it.

## Model Results

The current YOLO11n training run used:

- Epochs: 50
- Image size: 640
- Batch size: 8

Reported validation metrics from this run:

| Metric | Result |
| --- | --- |
| Precision | 0.382 |
| Recall | 0.297 |
| mAP@50 | 0.285 |
| mAP@50–95 | 0.155 |

These metrics reflect the current prepared subset and checkpoint, not production-level performance. Class imbalance, limited data, detection threshold, and dataset splits can affect results.

The trained checkpoint is named `best.pt`. In the development environment it was saved under the user’s `runs/detect/...` directory, outside the project root. Set `MODEL_PATH` to the actual checkpoint location on your machine.

## Project Structure

```
Scene Understanding System/
├── app/
│   ├── app.py
│   ├── components.py
│   └── styles.css
├── dataset/
│   ├── data.yaml
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
├── models/
│   └── yolov11/
├── src/
│   ├── detection/
│   │   └── detector.py
│   ├── scene/
│   │   └── scene_graph.py
│   ├── risk/
│   │   └── risk_analyzer.py
│   ├── nlp/
│   │   ├── vlm.py
│   │   └── report_generator.py
│   └── visualization/
│       └── visualizer.py
├── scripts/
│   └── prepare_bdd100k.py
├── outputs/
│   ├── reports/
│   └── visualizations/
├── tests/
├── run_detection.py
├── run_batch_analysis.py
├── run_risk_analysis.py
├── run_vlm_analysis.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
└── README.md
```

Actual files may differ slightly as development continues.

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd "Scene Understanding System"
```

Replace the placeholder with your repository URL.

### 2. Create an environment

Using Conda:

```bash
conda create -n road-scene python=3.10
conda activate road-scene
```

### 3. Install PyTorch

Choose the installation command for your OS and CUDA setup from the official selector: https://pytorch.org/get-started/locally/

For CPU-only use, install the CPU build. Avoid reinstalling a working CUDA-enabled PyTorch installation without a reason.

### 4. Install dependencies

If `requirements.txt` is complete and compatible with your environment:

```bash
pip install -r requirements.txt
```

Core packages used by the project include:

```bash
pip install ultralytics transformers accelerate sentencepiece streamlit opencv-python pillow pyyaml tqdm ijson
```

Resolve dependency conflicts in the intended project environment. The development environment previously reported a Streamlit/protobuf version conflict, so verify the app starts after installation.

### 5. Add the trained model

Place the checkpoint in a project-relative location, for example:

```
models/yolov11/best.pt
```

Or update `MODEL_PATH` in `app/app.py` and the run scripts to point to your actual checkpoint. Do not commit large model files unless you have checked repository size limits and redistribution permissions.

## Configuration

### Model path

The current development scripts use a Windows-specific absolute path. Update it for your computer:

```python
MODEL_PATH = r"C:\path\to\your\best.pt"
```

For portability, consider making the path project-relative or configurable through an environment variable.

### Confidence and detection limit

The dashboard defaults to confidence `0.60` and a maximum of `80` detections. Lower confidence can increase detections and false positives; higher confidence can miss objects. The detection cap helps limit downstream pairwise scene-graph computation.

## Run the Project

Run these commands from the project root using the environment where dependencies are installed.

### Object detection

```bash
python run_detection.py
```

### Single-image risk analysis

```bash
python run_risk_analysis.py
```

### VLM caption and report

```bash
python run_vlm_analysis.py
```

The first BLIP run may download model files from Hugging Face. Later runs generally use the local cache.

### Batch analysis

```bash
python run_batch_analysis.py
```

Review the script’s `NUM_IMAGES` setting before running.

### Streamlit dashboard

```bash
streamlit run app/app.py
```

Open the local URL printed in the terminal, typically `http://localhost:8501`. Upload a JPG, JPEG, or PNG road image and select **Analyze Road Scene**.

## Outputs

Depending on the script, outputs are saved under:

```
outputs/
├── reports/
└── visualizations/
```

Expected artifacts include annotated images, risk results, detected-class counts, spatial-relationship counts, BLIP caption, and a text report. Check each script’s console output for the exact output path.

## Risk Score Interpretation

The risk analyzer uses hand-crafted rules and a heuristic score. It considers selected pedestrian/rider/vehicle proximity or overlap patterns, nearby vehicle pairs, overlapping vehicle pairs, and scene density.

| Score | Level |
| --- | --- |
| 0–9 | SAFE |
| 10–34 | LOW |
| 35–69 | MEDIUM |
| 70–100 | HIGH |

These are implementation thresholds, not calibrated probabilities.

- A score of `28/100` does **not** mean a 28% chance of a crash.
- Bounding-box overlap does not prove a collision; occlusion and detector errors can cause overlap.
- `near` is based on 2D image geometry, not real-world distance.
- `left_of`, `above`, and `below` refer to image coordinates, not road topology or direction of travel.
- Detecting a traffic light does not identify its red/yellow/green state.
- This prototype must not be used for real-time driving decisions or safety-critical applications.

## Limitations

1. The current detector’s validation performance is modest; false positives and missed objects are possible.
2. The prepared dataset subset is relatively small and class distribution is uneven.
3. Scene-graph relations use 2D bounding boxes and simple geometric rules.
4. The risk score has not been calibrated against real accident outcomes.
5. BLIP generates a general image caption; it does not independently verify detections or calculate collision risk.
6. Runtime and memory use depend on image size, detection count, GPU availability, and configuration.
7. Absolute Windows paths must be updated before running on another machine.

## Troubleshooting

### `ModuleNotFoundError: No module named 'src'`

Run Streamlit from the project root and check that `src/` exists. The app should add the project root to `sys.path` before importing project modules.

### Checkpoint not found

Update `MODEL_PATH` in the dashboard and relevant scripts to the actual `best.pt` path.

### CUDA unavailable

Verify the active environment’s PyTorch build, NVIDIA driver, and CUDA compatibility. CPU inference may work but will generally be slower.

### Hugging Face warnings

Unauthenticated Hub and Windows symlink warnings do not necessarily mean the download failed. If the model loads and inference completes, these warnings can generally be left alone.

### Streamlit/protobuf conflict

Check the full traceback and installed package versions. Resolve the mismatch in the project environment instead of unnecessarily changing a working PyTorch/CUDA setup.

### Memory use is too high

- Process one image at a time.
- Keep the detection cap enabled.
- Avoid passing a whole directory to code designed for a single image.
- Close other GPU-heavy applications or reduce image size where supported.

## Future Improvements

- Train with more data and improve class balance.
- Evaluate per-class precision, recall, mAP, and confusion matrix on a documented test split.
- Add object tracking and video analysis.
- Improve scene-graph relation filtering and visualization.
- Validate and calibrate risk rules against a suitable evaluation dataset.
- Evaluate a stronger road-scene VLM or visual question-answering model within available hardware limits.
- Add JSON export, image history, and more downloadable artifacts.
- Add automated unit and integration tests.
- Make model paths portable and validate Docker setup.
- Add screenshots, CI checks, and reproducible experiment details.

## License

Choose and add a project license before publishing. Review the licenses and terms for BDD100K, Ultralytics, PyTorch, Transformers, BLIP model weights, and other third-party components. Do not imply unrestricted reuse until the relevant terms have been checked.

---

Built as an AI/ML project exploring road-scene perception, spatial reasoning, heuristic risk analysis, and vision-language reporting.
