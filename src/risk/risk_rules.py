# risk rules placeholder
"""
Risk rules for road-scene risk analysis.

The rules are based on 2D spatial relationships from the scene graph.
They do not claim true 3D distance or actual collision prediction.
"""


# Object categories
PEDESTRIAN = "pedestrian"
RIDER = "rider"
MOTORCYCLE = "motorcycle"
BICYCLE = "bicycle"

CAR = "car"
TRUCK = "truck"
BUS = "bus"


VEHICLES = {
    CAR,
    TRUCK,
    BUS,
    MOTORCYCLE,
    BICYCLE,
    RIDER,
}


VULNERABLE_ROAD_USERS = {
    PEDESTRIAN,
    RIDER,
    MOTORCYCLE,
    BICYCLE,
}


def get_risk_rule(class_a, class_b, relation):
    """
    Return a risk rule for a pair of objects and their relationship.

    Returns:
        dict or None
    """

    objects = {class_a, class_b}

    # ---------------------------------------------------------
    # 1. Pedestrian near vehicle
    # ---------------------------------------------------------
    if relation == "near":

        if PEDESTRIAN in objects and (
            CAR in objects
            or TRUCK in objects
            or BUS in objects
        ):
            return {
                "risk_type": "pedestrian_vehicle_proximity",
                "severity": "HIGH",
                "score": 30,
                "reason": (
                    "A pedestrian is in close proximity to a vehicle."
                ),
            }

        # -----------------------------------------------------
        # 2. Rider near vehicle
        # -----------------------------------------------------
        if RIDER in objects and (
            CAR in objects
            or TRUCK in objects
            or BUS in objects
        ):
            return {
                "risk_type": "rider_vehicle_proximity",
                "severity": "HIGH",
                "score": 25,
                "reason": (
                    "A rider is in close proximity to a vehicle."
                ),
            }

        # -----------------------------------------------------
        # 3. Motorcycle near vehicle
        # -----------------------------------------------------
        if MOTORCYCLE in objects and (
            CAR in objects
            or TRUCK in objects
            or BUS in objects
        ):
            return {
                "risk_type": "motorcycle_vehicle_proximity",
                "severity": "HIGH",
                "score": 25,
                "reason": (
                    "A motorcycle is in close proximity to another vehicle."
                ),
            }

        # -----------------------------------------------------
        # 4. Bicycle near vehicle
        # -----------------------------------------------------
        if BICYCLE in objects and (
            CAR in objects
            or TRUCK in objects
            or BUS in objects
        ):
            return {
                "risk_type": "bicycle_vehicle_proximity",
                "severity": "MEDIUM",
                "score": 20,
                "reason": (
                    "A bicycle is in close proximity to a vehicle."
                ),
            }

        # -----------------------------------------------------
        # 5. Pedestrian near motorcycle/bicycle/rider
        # -----------------------------------------------------
        if PEDESTRIAN in objects and (
            MOTORCYCLE in objects
            or BICYCLE in objects
            or RIDER in objects
        ):
            return {
                "risk_type": "vulnerable_user_proximity",
                "severity": "MEDIUM",
                "score": 15,
                "reason": (
                    "Two vulnerable road users are in close proximity."
                ),
            }

        # -----------------------------------------------------
        # 6. Vehicle near vehicle
        # -----------------------------------------------------
        if class_a in {CAR, TRUCK, BUS} and class_b in {
            CAR,
            TRUCK,
            BUS,
        }:
            return {
                "risk_type": "vehicle_proximity",
                "severity": "MEDIUM",
                "score": 10,
                "reason": (
                    "Multiple vehicles are in close spatial proximity."
                ),
            }

    # ---------------------------------------------------------
    # 7. Overlapping objects
    # ---------------------------------------------------------
    if relation == "overlapping":

        if PEDESTRIAN in objects:
            return {
                "risk_type": "pedestrian_overlap",
                "severity": "HIGH",
                "score": 35,
                "reason": (
                    "A pedestrian bounding box overlaps with another "
                    "road object."
                ),
            }

        if RIDER in objects or MOTORCYCLE in objects:
            return {
                "risk_type": "rider_overlap",
                "severity": "HIGH",
                "score": 30,
                "reason": (
                    "A rider or motorcycle overlaps with another "
                    "road object."
                ),
            }

        if BICYCLE in objects:
            return {
                "risk_type": "bicycle_overlap",
                "severity": "HIGH",
                "score": 25,
                "reason": (
                    "A bicycle overlaps with another road object."
                ),
            }

        if class_a in {CAR, TRUCK, BUS} and class_b in {
            CAR,
            TRUCK,
            BUS,
        }:
            return {
                "risk_type": "vehicle_overlap",
                "severity": "MEDIUM",
                "score": 20,
                "reason": (
                    "Two vehicle detections have overlapping "
                    "bounding boxes."
                ),
            }

    return None
