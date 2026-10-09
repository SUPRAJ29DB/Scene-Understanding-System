from ultralytics import YOLO


class RoadObjectDetector:

    def __init__(self, model_path, device=0):

        self.device = device

        self.model = YOLO(model_path)

        print(
            f"YOLO11 device: CUDA:{device}"
        )

    def detect(
        self,
        image_path,
        confidence=0.60
    ):

        results = self.model.predict(
            source=image_path,
            conf=confidence,
            iou=0.50,
            device=self.device,
            save=True,
            verbose=True
        )

        return results