class RiskAnalyzer:
    """
    Rule-based road-scene risk analyzer.

    The analyzer uses 2D spatial relationships from the scene graph.

    Important limitations:
    - 2D proximity does not represent real-world distance.
    - Bounding-box overlap does not prove a collision.
    - Traffic density alone is not considered a dangerous event.
    """

    def __init__(self):
        pass

    # =====================================================
    # MAIN ANALYSIS
    # =====================================================

    def analyze(self, scene_graph):

        nodes = scene_graph.get("nodes", [])
        edges = scene_graph.get("edges", [])

        risk_events = []

        # -------------------------------------------------
        # OBJECT GROUPS
        # -------------------------------------------------

        vehicles = [
            node for node in nodes
            if self._is_vehicle(node["class_name"])
        ]

        pedestrians = [
            node for node in nodes
            if node["class_name"] == "pedestrian"
        ]

        riders = [
            node for node in nodes
            if self._is_rider(node["class_name"])
        ]

        # =================================================
        # HIGH-RISK HUMAN INTERACTIONS
        # =================================================

        human_vehicle_pairs = set()
        human_rider_pairs = set()

        for edge in edges:

            source_id = edge["source"]
            target_id = edge["target"]
            relation = edge["relation"]

            source = self._find_node(
                nodes,
                source_id
            )

            target = self._find_node(
                nodes,
                target_id
            )

            if source is None or target is None:
                continue

            source_class = source["class_name"]
            target_class = target["class_name"]

            # -------------------------------------------------
            # PEDESTRIAN + VEHICLE
            # -------------------------------------------------

            if (
                self._is_vehicle(source_class)
                and target_class == "pedestrian"
            ) or (
                self._is_vehicle(target_class)
                and source_class == "pedestrian"
            ):

                pair = tuple(
                    sorted(
                        [source_id, target_id]
                    )
                )

                if pair in human_vehicle_pairs:
                    continue

                human_vehicle_pairs.add(pair)

                if relation == "near":

                    risk_events.append({
                        "risk_type":
                            "pedestrian_vehicle_proximity",

                        "severity":
                            "HIGH",

                        "score":
                            30,

                        "reason":
                            "A pedestrian appears spatially "
                            "close to a vehicle."
                    })

                elif relation == "overlapping":

                    risk_events.append({
                        "risk_type":
                            "pedestrian_vehicle_overlap",

                        "severity":
                            "HIGH",

                        "score":
                            45,

                        "reason":
                            "A pedestrian and vehicle have "
                            "overlapping bounding boxes. "
                            "This indicates a possible conflict "
                            "but does not prove a collision."
                    })

            # -------------------------------------------------
            # RIDER + VEHICLE
            # -------------------------------------------------

            if (
                self._is_rider(source_class)
                and self._is_vehicle(target_class)
            ) or (
                self._is_rider(target_class)
                and self._is_vehicle(source_class)
            ):

                pair = tuple(
                    sorted(
                        [source_id, target_id]
                    )
                )

                if pair in human_rider_pairs:
                    continue

                human_rider_pairs.add(pair)

                if relation == "near":

                    risk_events.append({
                        "risk_type":
                            "rider_vehicle_proximity",

                        "severity":
                            "HIGH",

                        "score":
                            25,

                        "reason":
                            "A rider or two-wheeler appears "
                            "close to a vehicle."
                    })

                elif relation == "overlapping":

                    risk_events.append({
                        "risk_type":
                            "rider_vehicle_overlap",

                        "severity":
                            "HIGH",

                        "score":
                            40,

                        "reason":
                            "A rider or two-wheeler has "
                            "an overlapping bounding box "
                            "with a vehicle."
                    })

            # -------------------------------------------------
            # PEDESTRIAN + RIDER
            # -------------------------------------------------

            if (
                source_class == "pedestrian"
                and self._is_rider(target_class)
            ) or (
                target_class == "pedestrian"
                and self._is_rider(source_class)
            ):

                if relation == "near":

                    risk_events.append({
                        "risk_type":
                            "pedestrian_rider_proximity",

                        "severity":
                            "HIGH",

                        "score":
                            25,

                        "reason":
                            "A pedestrian appears close to "
                            "a rider or two-wheeler."
                    })

                elif relation == "overlapping":

                    risk_events.append({
                        "risk_type":
                            "pedestrian_rider_overlap",

                        "severity":
                            "HIGH",

                        "score":
                            40,

                        "reason":
                            "A pedestrian and rider have "
                            "overlapping bounding boxes."
                    })

        # =====================================================
        # VEHICLE-VEHICLE INTERACTION
        # =====================================================
        #
        # IMPORTANT:
        # Normal traffic proximity should NOT dominate risk.
        #
        # We therefore:
        #   - count vehicle proximity as a small context signal
        #   - count overlap more seriously
        #   - aggregate all vehicle proximity into ONE event
        # =====================================================

        vehicle_near_pairs = set()
        vehicle_overlap_pairs = set()

        for edge in edges:

            source_id = edge["source"]
            target_id = edge["target"]
            relation = edge["relation"]

            source = self._find_node(
                nodes,
                source_id
            )

            target = self._find_node(
                nodes,
                target_id
            )

            if source is None or target is None:
                continue

            source_class = source["class_name"]
            target_class = target["class_name"]

            if not (
                self._is_vehicle(source_class)
                and self._is_vehicle(target_class)
            ):
                continue

            pair = tuple(
                sorted(
                    [source_id, target_id]
                )
            )

            # -------------------------------------------------
            # VEHICLE PROXIMITY
            # -------------------------------------------------

            if relation == "near":

                vehicle_near_pairs.add(pair)

            # -------------------------------------------------
            # VEHICLE OVERLAP
            # -------------------------------------------------

            elif relation == "overlapping":

                vehicle_overlap_pairs.add(pair)

        # =====================================================
        # AGGREGATED VEHICLE PROXIMITY
        # =====================================================

        if vehicle_near_pairs:

            near_count = len(
                vehicle_near_pairs
            )

            # Small contextual contribution.
            #
            # 1-2 nearby pairs -> 2 points
            # 3-5 nearby pairs -> 4 points
            # 6+ pairs          -> 6 points
            #

            if near_count <= 2:

                proximity_score = 2

            elif near_count <= 5:

                proximity_score = 4

            else:

                proximity_score = 6

            risk_events.append({

                "risk_type":
                    "vehicle_traffic_density",

                "severity":
                    "LOW",

                "score":
                    proximity_score,

                "reason":
                    f"{near_count} vehicle pair(s) "
                    "appear spatially close. This may "
                    "represent normal traffic density "
                    "rather than an actual hazardous event."
            })

        # =====================================================
        # VEHICLE OVERLAP
        # =====================================================

        if vehicle_overlap_pairs:

            overlap_count = len(
                vehicle_overlap_pairs
            )

            # Do not multiply risk excessively.
            #
            # One possible overlap = 15
            # Multiple overlaps = maximum 25
            #

            overlap_score = min(
                15 + (
                    (overlap_count - 1) * 5
                ),
                25
            )

            risk_events.append({

                "risk_type":
                    "possible_vehicle_conflict",

                "severity":
                    "MEDIUM",

                "score":
                    overlap_score,

                "reason":
                    f"{overlap_count} vehicle pair(s) "
                    "have overlapping bounding boxes. "
                    "This may indicate a possible spatial "
                    "conflict, but does not prove a collision."
            })

        # =====================================================
        # SCENE DENSITY
        # =====================================================

        object_count = len(nodes)

        # Density is contextual, not automatically dangerous.

        if object_count >= 15:

            risk_events.append({

                "risk_type":
                    "high_scene_density",

                "severity":
                    "LOW",

                "score":
                    5,

                "reason":
                    "Many road objects are visible in "
                    "the scene, indicating a relatively "
                    "dense traffic environment."
            })

        elif object_count >= 10:

            risk_events.append({

                "risk_type":
                    "moderate_scene_density",

                "severity":
                    "LOW",

                "score":
                    2,

                "reason":
                    "Several road objects are visible "
                    "in the scene."
            })

        # =====================================================
        # CALCULATE SCORE
        # =====================================================

        raw_score = sum(
            event["score"]
            for event in risk_events
        )

        risk_score = min(
            raw_score,
            100
        )

        # =====================================================
        # DETERMINE RISK LEVEL
        # =====================================================

        risk_level = self._get_risk_level(
            risk_score
        )

        # =====================================================
        # SUMMARY
        # =====================================================

        summary = self._generate_summary(
            risk_level,
            risk_score,
            risk_events,
            object_count
        )

        return {

            "risk_level":
                risk_level,

            "risk_score":
                risk_score,

            "object_count":
                object_count,

            "risk_events":
                risk_events,

            "summary":
                summary
        }

    # =====================================================
    # HELPER FUNCTIONS
    # =====================================================

    @staticmethod
    def _find_node(nodes, node_id):

        for node in nodes:

            if node["id"] == node_id:
                return node

        return None

    # -----------------------------------------------------

    @staticmethod
    def _is_vehicle(class_name):

        return class_name in {
            "car",
            "truck",
            "bus",
            "train"
        }

    # -----------------------------------------------------

    @staticmethod
    def _is_rider(class_name):

        return class_name in {
            "rider",
            "motorcycle",
            "bicycle"
        }

    # -----------------------------------------------------

    @staticmethod
    def _get_risk_level(score):

        if score >= 70:
            return "HIGH"

        elif score >= 35:
            return "MEDIUM"

        elif score >= 10:
            return "LOW"

        else:
            return "SAFE"

    # -----------------------------------------------------

    @staticmethod
    def _generate_summary(
        risk_level,
        risk_score,
        risk_events,
        object_count
    ):

        if not risk_events:

            return (
                "No significant visual risk indicators "
                "were detected in the road scene."
            )

        high_events = sum(
            1
            for event in risk_events
            if event["severity"] == "HIGH"
        )

        medium_events = sum(
            1
            for event in risk_events
            if event["severity"] == "MEDIUM"
        )

        low_events = sum(
            1
            for event in risk_events
            if event["severity"] == "LOW"
        )

        if risk_level == "HIGH":

            return (
                f"Potentially hazardous road scene detected. "
                f"Risk score: {risk_score}/100. "
                f"The scene contains {high_events} "
                f"high-severity visual risk indicator(s) "
                f"and {medium_events} medium-severity "
                f"indicator(s)."
            )

        elif risk_level == "MEDIUM":

            return (
                f"Moderate visual risk detected. "
                f"Risk score: {risk_score}/100. "
                f"The scene contains {medium_events} "
                f"medium-severity and {low_events} "
                f"low-severity visual indicator(s)."
            )

        elif risk_level == "LOW":

            return (
                f"Low visual risk detected. "
                f"Risk score: {risk_score}/100. "
                f"The scene mainly contains minor "
                f"spatial or traffic-density indicators."
            )

        else:

            return (
                f"No significant visual risk detected. "
                f"Risk score: {risk_score}/100. "
                f"{object_count} road object(s) "
                f"were identified."
            )