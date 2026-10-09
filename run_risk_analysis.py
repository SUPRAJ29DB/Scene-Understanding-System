from pathlib import Path
import torch

from src.detection.detector import RoadObjectDetector
from src.scene.scene_graph import SceneGraphBuilder
from src.risk.risk_analyzer import RiskAnalyzer
from src.visualization.visualizer import RoadSceneVisualizer


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_PATH = (
    r"C:\Users\Suprakash Ghosh\runs\detect\models"
    r"\yolov11\bdd100k_final-3\weights\best.pt"
)

VAL_DIR = Path(
    r"C:\Users\Suprakash Ghosh\Desktop"
    r"\Scene Understanding System\dataset\images\val"
)

# YOLO confidence threshold
CONFIDENCE_THRESHOLD = 0.60

# Maximum number of detections passed to scene graph
MAX_DETECTIONS = 80


# =========================================================
# GPU CHECK
# =========================================================

print("\n========================================")
print("             GPU CHECK")
print("========================================")

CUDA_AVAILABLE = torch.cuda.is_available()

print(
    f"CUDA available : {CUDA_AVAILABLE}"
)

if CUDA_AVAILABLE:

    print(
        "GPU            :",
        torch.cuda.get_device_name(0)
    )

    print(
        "CUDA version   :",
        torch.version.cuda
    )

else:

    print(
        "WARNING: CUDA is not available."
    )

    print(
        "YOLO11 may run on CPU."
    )

print("========================================")


# =========================================================
# SELECT ONE TEST IMAGE
# =========================================================

print("\n========================================")
print("        SELECTING TEST IMAGE")
print("========================================")


if not VAL_DIR.exists():

    raise FileNotFoundError(
        f"Validation directory does not exist:\n"
        f"{VAL_DIR}"
    )


image_files = sorted(
    list(VAL_DIR.glob("*.jpg")) +
    list(VAL_DIR.glob("*.jpeg")) +
    list(VAL_DIR.glob("*.png"))
)


if not image_files:

    raise FileNotFoundError(
        f"No image files found in:\n"
        f"{VAL_DIR}"
    )


# ---------------------------------------------------------
# Automatically select the first image
# ---------------------------------------------------------

IMAGE_PATH = image_files[0]


print(
    f"Selected image : "
    f"{IMAGE_PATH.name}"
)

print(
    f"Full path      : "
    f"{IMAGE_PATH}"
)

print("========================================")


# =========================================================
# ROAD SCENE ANALYSIS
# =========================================================

print("\n========================================")
print("        ROAD SCENE ANALYSIS")
print("========================================")


# =========================================================
# 1. YOLO11 OBJECT DETECTION
# =========================================================

print("\n[1/4] Running YOLO11 detection...")

print(
    f"Confidence threshold : "
    f"{CONFIDENCE_THRESHOLD}"
)

print(
    f"Maximum detections   : "
    f"{MAX_DETECTIONS}"
)


# ---------------------------------------------------------
# Initialize detector
# ---------------------------------------------------------

detector = RoadObjectDetector(
    MODEL_PATH
)


# ---------------------------------------------------------
# Run YOLO11
# ---------------------------------------------------------

results = detector.detect(
    str(IMAGE_PATH),
    confidence=CONFIDENCE_THRESHOLD
)


# ---------------------------------------------------------
# Safety check
# ---------------------------------------------------------

if not results:

    raise RuntimeError(
        "YOLO11 returned no results."
    )


# =========================================================
# 2. CONVERT YOLO RESULTS
# =========================================================

print("\nConverting YOLO results...")


detections = []


for result in results:

    boxes = result.boxes

    if boxes is None:
        continue


    for i in range(len(boxes)):

        class_id = int(
            boxes.cls[i]
        )

        confidence = float(
            boxes.conf[i]
        )

        bbox = boxes.xyxy[i].tolist()

        class_name = result.names[
            class_id
        ]


        detections.append({

            "class_name": class_name,

            "confidence": confidence,

            "bbox": bbox

        })


print(
    f"Raw detections : "
    f"{len(detections)}"
)


# =========================================================
# 3. LIMIT DETECTIONS
# =========================================================

if len(detections) > MAX_DETECTIONS:

    print(
        f"\nWARNING: "
        f"{len(detections)} detections found."
    )

    print(
        f"Keeping top "
        f"{MAX_DETECTIONS} detections "
        f"by confidence."
    )


    detections = sorted(
        detections,
        key=lambda x: x["confidence"],
        reverse=True
    )[:MAX_DETECTIONS]


else:

    print(
        "\nDetections are within "
        "the safe limit."
    )


print(
    f"Detections passed to scene graph : "
    f"{len(detections)}"
)


# =========================================================
# 4. DISPLAY DETECTED OBJECTS
# =========================================================

print("\nDetected Objects:")


if not detections:

    print(
        "  No objects detected."
    )

else:

    for i, detection in enumerate(
        detections,
        start=1
    ):

        print(
            f"  {i:02d}. "
            f"{detection['class_name']:<18} "
            f"confidence="
            f"{detection['confidence']:.2f}"
        )


# =========================================================
# 5. GET IMAGE DIMENSIONS
# =========================================================

image_width = (
    results[0].orig_shape[1]
)

image_height = (
    results[0].orig_shape[0]
)


print(
    f"\nImage size: "
    f"{image_width} x {image_height}"
)


# =========================================================
# 6. BUILD SCENE GRAPH
# =========================================================

print("\n[2/4] Building scene graph...")

print(
    f"Processing "
    f"{len(detections)} objects..."
)


scene_builder = SceneGraphBuilder()


scene_graph = scene_builder.build(
    detections=detections,
    image_width=image_width,
    image_height=image_height
)


node_count = len(
    scene_graph["nodes"]
)

edge_count = len(
    scene_graph["edges"]
)


print(
    f"Scene graph created: "
    f"{node_count} nodes, "
    f"{edge_count} relationships."
)


# =========================================================
# 7. RISK ANALYSIS
# =========================================================

print("\n[3/4] Running risk analysis...")


risk_analyzer = RiskAnalyzer()


risk_result = risk_analyzer.analyze(
    scene_graph
)


# =========================================================
# 8. DISPLAY RISK RESULT
# =========================================================

print("\n========================================")
print("        FINAL RISK ANALYSIS")
print("========================================")


print(
    f"Risk Level   : "
    f"{risk_result['risk_level']}"
)


print(
    f"Risk Score   : "
    f"{risk_result['risk_score']}/100"
)


print(
    f"Object Count : "
    f"{risk_result['object_count']}"
)


print("\nSummary:")


print(
    risk_result["summary"]
)


# =========================================================
# 9. DISPLAY RISK EVENTS
# =========================================================

print("\nRisk Events:")


if not risk_result["risk_events"]:

    print(
        "No significant risk detected."
    )

else:

    for i, event in enumerate(
        risk_result["risk_events"],
        start=1
    ):

        print(
            f"\n{i}. "
            f"{event['risk_type']}"
        )

        print(
            f"   Severity : "
            f"{event['severity']}"
        )

        print(
            f"   Score    : "
            f"{event['score']}"
        )

        print(
            f"   Reason   : "
            f"{event['reason']}"
        )


# =========================================================
# 10. VISUALIZATION
# =========================================================

print("\n[4/4] Creating visualization...")


# ---------------------------------------------------------
# Create output directory
# ---------------------------------------------------------

OUTPUT_DIR = Path(
    "outputs"
) / "visualizations"


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# Output filename
# ---------------------------------------------------------

OUTPUT_PATH = (
    OUTPUT_DIR
    / f"{IMAGE_PATH.stem}_analysis.jpg"
)


# ---------------------------------------------------------
# Initialize visualizer
# ---------------------------------------------------------

visualizer = RoadSceneVisualizer()


# ---------------------------------------------------------
# Generate annotated image
# ---------------------------------------------------------

saved_image = visualizer.visualize(

    image_path=IMAGE_PATH,

    detections=detections,

    risk_result=risk_result,

    output_path=OUTPUT_PATH,
    scene_graph=scene_graph
)


print(
    "\nVisualization saved successfully:"
)

print(
    saved_image
)


# =========================================================
# 11. DISPLAY SCENE GRAPH
# =========================================================

print("\n========================================")
print("        SCENE GRAPH")
print("========================================")


# ---------------------------------------------------------
# Nodes
# ---------------------------------------------------------

print("\nNodes:")


for node in scene_graph["nodes"]:

    print(
        f"  {node['id']} | "
        f"{node['class_name']} | "
        f"confidence="
        f"{node['confidence']:.2f}"
    )


# ---------------------------------------------------------
# Relationships
# ---------------------------------------------------------

print("\nRelationships:")


for edge in scene_graph["edges"]:

    print(
        f"  {edge['source']} "
        f"--[{edge['relation']}]--> "
        f"{edge['target']}"
    )


# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n========================================")
print("        ANALYSIS COMPLETED")
print("========================================")


print(
    f"\nProcessed image : "
    f"{IMAGE_PATH.name}"
)


print(
    f"Objects         : "
    f"{node_count}"
)


print(
    f"Relationships    : "
    f"{edge_count}"
)


print(
    f"Risk level       : "
    f"{risk_result['risk_level']}"
)


print(
    f"Risk score       : "
    f"{risk_result['risk_score']}/100"
)


print(
    f"Visualization    : "
    f"{saved_image}"
)


print("\n========================================")