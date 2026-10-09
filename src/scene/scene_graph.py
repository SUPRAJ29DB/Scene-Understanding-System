# scene graph placeholder
import math


class SceneGraphBuilder:

    def __init__(self, near_threshold=0.18):
        self.near_threshold = near_threshold

    @staticmethod
    def get_center(bbox):
        x1, y1, x2, y2 = bbox

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        return center_x, center_y

    @staticmethod
    def calculate_iou(box1, box2):

        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])

        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection_width = max(0, x2 - x1)
        intersection_height = max(0, y2 - y1)

        intersection_area = (
            intersection_width * intersection_height
        )

        area1 = (
            (box1[2] - box1[0]) *
            (box1[3] - box1[1])
        )

        area2 = (
            (box2[2] - box2[0]) *
            (box2[3] - box2[1])
        )

        union_area = area1 + area2 - intersection_area

        if union_area == 0:
            return 0

        return intersection_area / union_area

    def build(self, detections, image_width, image_height):

        nodes = []
        edges = []

        # --------------------------------
        # Create nodes
        # --------------------------------

        for i, detection in enumerate(detections):

            bbox = detection["bbox"]

            center_x, center_y = self.get_center(bbox)

            node = {
                "id": f'{detection["class_name"]}_{i + 1}',
                "class_name": detection["class_name"],
                "confidence": detection["confidence"],
                "bbox": bbox,
                "center": [center_x, center_y]
            }

            nodes.append(node)

        # --------------------------------
        # Create relationships
        # --------------------------------

        for i in range(len(nodes)):

            for j in range(i + 1, len(nodes)):

                obj1 = nodes[i]
                obj2 = nodes[j]

                x1, y1 = obj1["center"]
                x2, y2 = obj2["center"]

                dx = (x2 - x1) / image_width
                dy = (y2 - y1) / image_height

                distance = math.sqrt(
                    dx ** 2 + dy ** 2
                )

                # ----------------------------
                # LEFT / RIGHT
                # ----------------------------

                if abs(dx) > 0.05:

                    if x1 < x2:
                        edges.append({
                            "source": obj1["id"],
                            "relation": "left_of",
                            "target": obj2["id"]
                        })
                    else:
                        edges.append({
                            "source": obj1["id"],
                            "relation": "left_of",
                            "target": obj2["id"]
                        })

                # ----------------------------
                # ABOVE / BELOW
                # ----------------------------

                if abs(dy) > 0.05:

                    if y1 < y2:
                        edges.append({
                            "source": obj1["id"],
                            "relation": "above",
                            "target": obj2["id"]
                        })
                    else:
                        edges.append({
                            "source": obj1["id"],
                            "relation": "below",
                            "target": obj2["id"]
                        })

                # ----------------------------
                # NEAR
                # ----------------------------

                if distance < self.near_threshold:

                    edges.append({
                        "source": obj1["id"],
                        "relation": "near",
                        "target": obj2["id"]
                    })

                # ----------------------------
                # OVERLAP
                # ----------------------------

                iou = self.calculate_iou(
                    obj1["bbox"],
                    obj2["bbox"]
                )

                if iou > 0.10:

                    edges.append({
                        "source": obj1["id"],
                        "relation": "overlapping",
                        "target": obj2["id"]
                    })

        return {
            "nodes": nodes,
            "edges": edges
        }