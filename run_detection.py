from src.detection.detector import RoadObjectDetector


MODEL_PATH = r"C:\Users\Suprakash Ghosh\runs\detect\models\yolov11\bdd100k_final-3\weights\best.pt"

IMAGE_PATH = r"C:\Users\Suprakash Ghosh\Desktop\Scene Understanding System\dataset\images\val"


detector = RoadObjectDetector(MODEL_PATH)

results = detector.detect(IMAGE_PATH)

print("\n========== DETECTION RESULTS ==========\n")

for result in results:

    boxes = result.boxes

    for i in range(len(boxes)):

        class_id = int(boxes.cls[i])
        confidence = float(boxes.conf[i])

        x1, y1, x2, y2 = boxes.xyxy[i].tolist()

        class_name = result.names[class_id]

        print(
            f"{i + 1}. "
            f"Object: {class_name:<15} "
            f"Confidence: {confidence:.2f} "
            f"Bounding Box: "
            f"({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"
        )

print("\n========================================")
print("Detection completed!")