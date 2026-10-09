from pathlib import Path
import torch

from src.detection.detector import RoadObjectDetector
from src.scene.scene_graph import SceneGraphBuilder
from src.risk.risk_analyzer import RiskAnalyzer
from src.nlp.vlm import RoadSceneVLM
from src.nlp.report_generator import RoadSceneReportGenerator


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = (
    r"C:\Users\Suprakash Ghosh\runs\detect\models"
    r"\yolov11\bdd100k_final-3\weights\best.pt"
)

VAL_DIR = Path(
    r"C:\Users\Suprakash Ghosh\Desktop"
    r"\Scene Understanding System\dataset\images\val"
)

CONFIDENCE_THRESHOLD = 0.60
MAX_DETECTIONS = 80


# ============================================================
# GPU CHECK
# ============================================================

print()
print("=" * 50)
print("              GPU CHECK")
print("=" * 50)

print(
    "CUDA available :",
    torch.cuda.is_available()
)

if torch.cuda.is_available():

    print(
        "GPU            :",
        torch.cuda.get_device_name(0)
    )

    print(
        "CUDA version   :",
        torch.version.cuda
    )

print("=" * 50)


# ============================================================
# SELECT IMAGE
# ============================================================

print()
print("=" * 50)
print("          SELECTING TEST IMAGE")
print("=" * 50)

image_files = sorted(
    list(VAL_DIR.glob("*.jpg")) +
    list(VAL_DIR.glob("*.jpeg")) +
    list(VAL_DIR.glob("*.png"))
)

if not image_files:

    raise FileNotFoundError(
        f"No images found in {VAL_DIR}"
    )

IMAGE_PATH = image_files[0]

print(
    "Selected image :",
    IMAGE_PATH.name
)

print(
    "Full path      :",
    IMAGE_PATH
)

print("=" * 50)


# ============================================================
# 1. YOLO11 DETECTION
# ============================================================

print()
print("=" * 50)
print("       [1/5] YOLO11 OBJECT DETECTION")
print("=" * 50)

detector = RoadObjectDetector(
    MODEL_PATH
)

results = detector.detect(
    str(IMAGE_PATH),
    confidence=CONFIDENCE_THRESHOLD
)

result = results[0]

detections = []

boxes = result.boxes

for i in range(len(boxes)):

    cls_id = int(
        boxes.cls[i].item()
    )

    confidence = float(
        boxes.conf[i].item()
    )

    x1, y1, x2, y2 = (
        boxes.xyxy[i]
        .tolist()
    )

    class_name = detector.model.names[
        cls_id
    ]

    detections.append(
        {
            "id": i + 1,
            "class_name": class_name,
            "confidence": confidence,
            "bbox": [
                x1,
                y1,
                x2,
                y2
            ]
        }
    )


# Limit detections
detections = sorted(
    detections,
    key=lambda x: x["confidence"],
    reverse=True
)[:MAX_DETECTIONS]

print(
    f"Detections passed forward: "
    f"{len(detections)}"
)

for i, detection in enumerate(
    detections,
    start=1
):

    print(
        f"  {i:02d}. "
        f"{detection['class_name']:<18}"
        f"confidence="
        f"{detection['confidence']:.2f}"
    )


# ============================================================
# 2. SCENE GRAPH
# ============================================================

print()
print("=" * 50)
print("       [2/5] BUILDING SCENE GRAPH")
print("=" * 50)

image_width = result.orig_shape[1]
image_height = result.orig_shape[0]

scene_builder = SceneGraphBuilder()

scene_graph = scene_builder.build(
    detections=detections,
    image_width=image_width,
    image_height=image_height
)

if isinstance(scene_graph, dict):

    nodes = scene_graph.get(
        "nodes",
        []
    )

    edges = scene_graph.get(
        "edges",
        []
    )

else:

    nodes = getattr(
        scene_graph,
        "nodes",
        []
    )

    edges = getattr(
        scene_graph,
        "edges",
        []
    )

print(
    f"Scene graph: "
    f"{len(nodes)} nodes, "
    f"{len(edges)} relationships"
)


# ============================================================
# 3. RISK ANALYSIS
# ============================================================

print()
print("=" * 50)
print("          [3/5] RISK ANALYSIS")
print("=" * 50)

risk_analyzer = RiskAnalyzer()

risk_result = risk_analyzer.analyze(
    scene_graph
)

risk_level = risk_result.get(
    "risk_level",
    "UNKNOWN"
)

risk_score = risk_result.get(
    "risk_score",
    0
)

print(
    "Risk Level :",
    risk_level
)

print(
    "Risk Score :",
    f"{risk_score}/100"
)


# ============================================================
# 4. VLM
# ============================================================

print()
print("=" * 50)
print("          [4/5] VLM ANALYSIS")
print("=" * 50)

vlm = RoadSceneVLM()

print(
    "Generating visual description..."
)

caption = vlm.generate_caption(
    IMAGE_PATH
)

print()
print("VLM Caption:")
print(caption)


# ============================================================
# 5. FINAL REPORT
# ============================================================

print()
print("=" * 50)
print("          [5/5] GENERATING REPORT")
print("=" * 50)

report_generator = (
    RoadSceneReportGenerator()
)

report = report_generator.generate_report(
    caption=caption,
    detections=detections,
    scene_graph=scene_graph,
    risk_result=risk_result
)

print()
print(report)


# ============================================================
# SAVE REPORT
# ============================================================

OUTPUT_DIR = Path(
    "outputs"
) / "reports"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_PATH = (
    OUTPUT_DIR /
    f"{IMAGE_PATH.stem}_report.txt"
)

REPORT_PATH.write_text(
    report,
    encoding="utf-8"
)

print()
print("=" * 50)
print("          ANALYSIS COMPLETED")
print("=" * 50)

print(
    "Report saved:",
    REPORT_PATH
)

print("=" * 50)