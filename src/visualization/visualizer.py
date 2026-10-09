from pathlib import Path
import cv2


class RoadSceneVisualizer:

    def visualize(
        self,
        image_path,
        detections,
        risk_result,
        output_path,
        scene_graph=None
    ):
        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(
                f"Could not load image: {image_path}"
            )

        # --------------------------------------------------
        # 1. Draw YOLO detections
        # --------------------------------------------------

        centers = {}

        for detection in detections:
            bbox = detection["bbox"]
            class_name = detection["class_name"]
            confidence = detection["confidence"]

            x1, y1, x2, y2 = map(int, bbox)

            # Bounding box
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Center point
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            detection_id = detection.get(
                "id",
                detection.get("track_id", None)
            )

            if detection_id is not None:
                centers[detection_id] = (cx, cy)

            label = f"{class_name} {confidence:.2f}"

            cv2.putText(
                image,
                label,
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )

        # --------------------------------------------------
        # 2. Draw Scene Graph Relationships
        # --------------------------------------------------

        if scene_graph is not None:

            # Support either:
            # scene_graph["edges"]
            # or scene_graph.edges
            if isinstance(scene_graph, dict):
                edges = scene_graph.get("edges", [])
                nodes = scene_graph.get("nodes", [])
            else:
                edges = getattr(scene_graph, "edges", [])
                nodes = getattr(scene_graph, "nodes", [])

            # Build centers from scene graph nodes
            for node in nodes:

                if isinstance(node, dict):
                    node_id = node.get("id")
                    center = node.get("center")
                else:
                    node_id = getattr(node, "id", None)
                    center = getattr(node, "center", None)

                if node_id is not None and center is not None:
                    centers[node_id] = (
                        int(center[0]),
                        int(center[1])
                    )

            relationship_count = 0

            # Draw only important relationships
            for edge in edges:

                if relationship_count >= 20:
                    break

                if isinstance(edge, dict):
                    source = edge.get("source")
                    target = edge.get("target")
                    relation = edge.get(
                        "relation",
                        edge.get("relationship")
                    )
                else:
                    source = getattr(edge, "source", None)
                    target = getattr(edge, "target", None)
                    relation = getattr(
                        edge,
                        "relation",
                        getattr(edge, "relationship", None)
                    )

                if source not in centers or target not in centers:
                    continue

                if relation is None:
                    continue

                relation = str(relation)

                # Only important spatial relationships
                if relation not in {
                    "near",
                    "overlapping",
                    "left_of",
                    "above",
                    "below"
                }:
                    continue

                p1 = centers[source]
                p2 = centers[target]

                # Relationship colors
                if relation == "near":
                    color = (0, 255, 255)       # Yellow

                elif relation == "overlapping":
                    color = (0, 0, 255)         # Red

                elif relation == "left_of":
                    color = (255, 0, 0)         # Blue

                else:
                    color = (255, 0, 255)       # Purple

                # Draw relationship line
                cv2.line(
                    image,
                    p1,
                    p2,
                    color,
                    2
                )

                # Arrow for directional relations
                if relation in {
                    "left_of",
                    "above",
                    "below"
                }:
                    cv2.arrowedLine(
                        image,
                        p1,
                        p2,
                        color,
                        2,
                        tipLength=0.15
                    )

                # Label position
                label_x = int((p1[0] + p2[0]) / 2)
                label_y = int((p1[1] + p2[1]) / 2)

                # Background rectangle
                (tw, th), _ = cv2.getTextSize(
                    relation,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    1
                )

                cv2.rectangle(
                    image,
                    (label_x - 2, label_y - th - 4),
                    (label_x + tw + 2, label_y + 3),
                    (0, 0, 0),
                    -1
                )

                cv2.putText(
                    image,
                    relation,
                    (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                    cv2.LINE_AA
                )

                relationship_count += 1

        # --------------------------------------------------
        # 3. Risk Information
        # --------------------------------------------------

        risk_level = risk_result.get(
            "risk_level",
            risk_result.get("level", "UNKNOWN")
        )

        risk_score = risk_result.get(
            "risk_score",
            risk_result.get("score", 0)
        )

        risk_events = risk_result.get(
            "events",
            risk_result.get("risk_events", [])
        )

        # --------------------------------------------------
        # 4. Title
        # --------------------------------------------------

        cv2.rectangle(
            image,
            (35, 95),
            (205, 115),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            image,
            "AI ROAD SCENE ANALYSIS",
            (40, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        # --------------------------------------------------
        # 5. Risk Panel
        # --------------------------------------------------

        panel_height = 65

        overlay = image.copy()

        cv2.rectangle(
            overlay,
            (
                0,
                image.shape[0] - panel_height
            ),
            (
                image.shape[1],
                image.shape[0]
            ),
            (20, 20, 20),
            -1
        )

        image = cv2.addWeighted(
            overlay,
            0.85,
            image,
            0.15,
            0
        )

        # Risk color
        if risk_level == "HIGH":
            risk_color = (0, 0, 255)

        elif risk_level == "MEDIUM":
            risk_color = (0, 165, 255)

        elif risk_level == "LOW":
            risk_color = (0, 255, 255)

        else:
            risk_color = (0, 255, 0)

        y = image.shape[0] - 43

        cv2.putText(
            image,
            f"RISK LEVEL: {risk_level}",
            (40, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            risk_color,
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            image,
            f"RISK SCORE: {risk_score}/100",
            (40, y + 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            image,
            f"OBJECTS: {len(detections)}",
            (40, y + 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            image,
            f"RISK EVENTS: {len(risk_events)}",
            (145, y + 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        # --------------------------------------------------
        # 6. Save
        # --------------------------------------------------

        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        success = cv2.imwrite(
            str(output_path),
            image
        )

        if not success:
            raise RuntimeError(
                f"Failed to save image: {output_path}"
            )

        return output_path