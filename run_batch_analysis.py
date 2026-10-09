from pathlib import Path
import torch

from src.detection.detector import RoadObjectDetector
from src.scene.scene_graph import SceneGraphBuilder
from src.risk.risk_analyzer import RiskAnalyzer


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

# Number of validation images to test
NUM_IMAGES = 10

# YOLO confidence threshold
CONFIDENCE = 0.60

# Maximum detections passed to scene graph
MAX_DETECTIONS = 80


# =========================================================
# GPU CHECK
# =========================================================

print("\n========================================")
print("             GPU CHECK")
print("========================================")

if torch.cuda.is_available():

    print("CUDA available : True")

    print(
        "GPU            :",
        torch.cuda.get_device_name(0)
    )

    print(
        "CUDA version   :",
        torch.version.cuda
    )

else:

    print("CUDA available : False")
    print("WARNING: YOLO11 will run on CPU.")

print("========================================")


# =========================================================
# CHECK VALIDATION DIRECTORY
# =========================================================

if not VAL_DIR.exists():

    raise FileNotFoundError(
        f"\nValidation directory not found:\n{VAL_DIR}"
    )


# =========================================================
# FIND IMAGES
# =========================================================

image_files = sorted(
    list(VAL_DIR.glob("*.jpg")) +
    list(VAL_DIR.glob("*.jpeg")) +
    list(VAL_DIR.glob("*.png"))
)


if not image_files:

    raise FileNotFoundError(
        f"\nNo images found in:\n{VAL_DIR}"
    )


# Only use the requested number of images
image_files = image_files[:NUM_IMAGES]


print("\n========================================")
print("        BATCH TEST CONFIGURATION")
print("========================================")

print(
    f"Images available : {len(image_files)}"
)

print(
    f"Images to test   : {len(image_files)}"
)

print(
    f"Confidence       : {CONFIDENCE}"
)

print(
    f"Max detections   : {MAX_DETECTIONS}"
)

print("========================================")


# =========================================================
# LOAD MODELS
# =========================================================

print("\nLoading YOLO11 model...")

detector = RoadObjectDetector(
    MODEL_PATH
)

scene_builder = SceneGraphBuilder()

risk_analyzer = RiskAnalyzer()


# =========================================================
# STORE RESULTS
# =========================================================

results_summary = []


# =========================================================
# PROCESS EACH IMAGE
# =========================================================

for image_number, image_path in enumerate(
    image_files,
    start=1
):

    print("\n\n")
    print("========================================")

    print(
        f"IMAGE {image_number}/{len(image_files)}"
    )

    print("========================================")

    print(
        f"Image: {image_path.name}"
    )


    try:

        # =================================================
        # 1. YOLO11 DETECTION
        # =================================================

        print("\n[1] YOLO11 detection...")

        yolo_results = detector.detect(
            str(image_path),
            confidence=CONFIDENCE
        )


        if not yolo_results:

            print(
                "No YOLO results returned."
            )

            continue


        # =================================================
        # 2. CONVERT YOLO RESULTS
        # =================================================

        detections = []


        for result in yolo_results:

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


        # =================================================
        # 3. LIMIT DETECTIONS
        # =================================================

        if len(detections) > MAX_DETECTIONS:

            print(
                f"Limiting detections "
                f"to {MAX_DETECTIONS}."
            )


            detections = sorted(
                detections,
                key=lambda x: x["confidence"],
                reverse=True
            )[:MAX_DETECTIONS]


        print(
            f"Final detections: "
            f"{len(detections)}"
        )


        # =================================================
        # 4. COUNT OBJECT CLASSES
        # =================================================

        class_counts = {}


        for detection in detections:

            class_name = detection[
                "class_name"
            ]

            class_counts[class_name] = (
                class_counts.get(
                    class_name,
                    0
                ) + 1
            )


        print("\nDetected objects:")


        if class_counts:

            for class_name, count in sorted(
                class_counts.items()
            ):

                print(
                    f"  {class_name:<18}: "
                    f"{count}"
                )

        else:

            print("  None")


        # =================================================
        # 5. IMAGE DIMENSIONS
        # =================================================

        image_width = (
            yolo_results[0].orig_shape[1]
        )

        image_height = (
            yolo_results[0].orig_shape[0]
        )


        print(
            f"\nImage size: "
            f"{image_width} x {image_height}"
        )


        # =================================================
        # 6. BUILD SCENE GRAPH
        # =================================================

        print(
            "\n[2] Building scene graph..."
        )


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
            f"Nodes         : "
            f"{node_count}"
        )

        print(
            f"Relationships : "
            f"{edge_count}"
        )


        # =================================================
        # 7. RISK ANALYSIS
        # =================================================

        print(
            "\n[3] Running risk analysis..."
        )


        risk_result = risk_analyzer.analyze(
            scene_graph
        )


        risk_level = risk_result[
            "risk_level"
        ]

        risk_score = risk_result[
            "risk_score"
        ]

        object_count = risk_result[
            "object_count"
        ]

        risk_event_count = len(
            risk_result["risk_events"]
        )


        print(
            f"Risk level  : "
            f"{risk_level}"
        )

        print(
            f"Risk score  : "
            f"{risk_score}/100"
        )

        print(
            f"Risk events : "
            f"{risk_event_count}"
        )


        # =================================================
        # 8. STORE RESULT
        # =================================================

        results_summary.append({

            "image": image_path.name,

            "objects": object_count,

            "relationships": edge_count,

            "risk_level": risk_level,

            "risk_score": risk_score,

            "risk_events": risk_event_count

        })


        print(
            "\n✓ Image completed successfully."
        )


    except Exception as e:

        print(
            "\n✗ ERROR processing image"
        )

        print(
            f"Error type: "
            f"{type(e).__name__}"
        )

        print(
            f"Error message: {e}"
        )

        print(
            "\nSkipping this image "
            "and continuing..."
        )


# =========================================================
# FINAL BATCH SUMMARY
# =========================================================

print("\n\n")
print("========================================")
print("        BATCH TEST SUMMARY")
print("========================================")


if not results_summary:

    print(
        "No images were processed successfully."
    )

else:

    print(
        f"Successfully processed: "
        f"{len(results_summary)}"
    )


    print("\nResults:\n")


    print(
        f"{'Image':<30}"
        f"{'Objects':<10}"
        f"{'Relations':<12}"
        f"{'Risk':<10}"
        f"{'Score':<10}"
    )

    print("-" * 72)


    for result in results_summary:

        print(
            f"{result['image']:<30}"
            f"{result['objects']:<10}"
            f"{result['relationships']:<12}"
            f"{result['risk_level']:<10}"
            f"{result['risk_score']:<10}"
        )


# =========================================================
# STATISTICS
# =========================================================

if results_summary:

    print("\n========================================")
    print("             STATISTICS")
    print("========================================")


    average_score = (
        sum(
            result["risk_score"]
            for result in results_summary
        )
        / len(results_summary)
    )


    print(
        f"Average risk score : "
        f"{average_score:.2f}/100"
    )


    # -----------------------------------------------------
    # Risk distribution
    # -----------------------------------------------------

    risk_distribution = {}


    for result in results_summary:

        level = result["risk_level"]

        risk_distribution[level] = (
            risk_distribution.get(
                level,
                0
            ) + 1
        )


    print("\nRisk distribution:")


    for level, count in sorted(
        risk_distribution.items()
    ):

        print(
            f"  {level:<8}: {count}"
        )


    # -----------------------------------------------------
    # Highest risk image
    # -----------------------------------------------------

    highest_risk = max(
        results_summary,
        key=lambda x: x["risk_score"]
    )


    print("\nHighest risk image:")

    print(
        f"  Image : "
        f"{highest_risk['image']}"
    )

    print(
        f"  Level : "
        f"{highest_risk['risk_level']}"
    )

    print(
        f"  Score : "
        f"{highest_risk['risk_score']}/100"
    )


print("\n========================================")
print("        BATCH TEST COMPLETED")
print("========================================")