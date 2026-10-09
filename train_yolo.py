from ultralytics import YOLO


def main():
    # Load pretrained YOLO11n
    model = YOLO("yolo11n.pt")

    # Train on BDD100K
    results = model.train(
        data="dataset/data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        workers=0,
        project="models/yolov11",
        name="bdd100k_final"
    )

    print("Training completed successfully!")


if __name__ == "__main__":
    main()