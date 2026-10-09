from pathlib import Path
import shutil
import ijson
from PIL import Image
from tqdm import tqdm


# ============================================================
# BDD100K LOCATION
# ============================================================

BDD_ROOT = Path(r"C:\Users\Suprakash Ghosh\Downloads\archive")

IMAGE_ROOT = (
    BDD_ROOT
    / "bdd100k"
    / "bdd100k"
    / "images"
    / "100k"
)

LABEL_ROOT = (
    BDD_ROOT
    / "bdd100k_labels_release"
    / "bdd100k"
    / "labels"
)


# ============================================================
# PROJECT LOCATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_IMAGE_ROOT = PROJECT_ROOT / "dataset" / "images"
OUTPUT_LABEL_ROOT = PROJECT_ROOT / "dataset" / "labels"


# ============================================================
# DATASET SIZE
# ============================================================

TRAIN_LIMIT = 8000
VAL_LIMIT = 2000


# ============================================================
# BDD100K CLASS MAPPING
# ============================================================

CLASS_MAP = {
    "pedestrian": 0,
    "rider": 1,
    "car": 2,
    "truck": 3,
    "bus": 4,
    "train": 5,
    "motorcycle": 6,
    "bicycle": 7,
    "traffic light": 8,
    "traffic sign": 9,
}


# ============================================================
# CONVERT BOUNDING BOX TO YOLO FORMAT
# ============================================================

def convert_box(box, image_width, image_height):
    x1 = box["x1"]
    y1 = box["y1"]
    x2 = box["x2"]
    y2 = box["y2"]

    x_center = ((x1 + x2) / 2) / image_width
    y_center = ((y1 + y2) / 2) / image_height

    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height

    return x_center, y_center, width, height


# ============================================================
# PROCESS ONE SPLIT
# ============================================================

def process_split(split, limit):
    json_file = (
        LABEL_ROOT
        / f"bdd100k_labels_images_{split}.json"
    )

    image_source = IMAGE_ROOT / split

    image_destination = OUTPUT_IMAGE_ROOT / split
    label_destination = OUTPUT_LABEL_ROOT / split

    image_destination.mkdir(parents=True, exist_ok=True)
    label_destination.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 60)
    print(f"Processing {split.upper()} dataset")
    print("=" * 60)

    if not json_file.exists():
        print(f"ERROR: Annotation file not found:")
        print(json_file)
        return

    if not image_source.exists():
        print(f"ERROR: Image folder not found:")
        print(image_source)
        return

    processed = 0

    with open(json_file, "rb") as f:

        for item in ijson.items(f, "item"):

            if processed >= limit:
                break

            image_name = item["name"]

            image_path = image_source / image_name

            if not image_path.exists():
                print(f"Warning: Image not found: {image_name}")
                continue

            # ------------------------------------------------
            # Read image dimensions
            # ------------------------------------------------

            try:
                with Image.open(image_path) as img:
                    image_width, image_height = img.size
            except Exception as e:
                print(f"Could not read {image_name}: {e}")
                continue

            # ------------------------------------------------
            # Create YOLO label file
            # ------------------------------------------------

            label_lines = []

            for label in item.get("labels", []):

                category = label.get("category")

                if category not in CLASS_MAP:
                    continue

                if "box2d" not in label:
                    continue

                class_id = CLASS_MAP[category]

                box = label["box2d"]

                x_center, y_center, width, height = convert_box(
                    box,
                    image_width,
                    image_height
                )

                label_lines.append(
                    f"{class_id} "
                    f"{x_center:.6f} "
                    f"{y_center:.6f} "
                    f"{width:.6f} "
                    f"{height:.6f}"
                )

            # ------------------------------------------------
            # Copy image
            # ------------------------------------------------

            destination_image = image_destination / image_name

            shutil.copy2(
                image_path,
                destination_image
            )

            # ------------------------------------------------
            # Save YOLO annotation
            # ------------------------------------------------

            label_name = Path(image_name).stem + ".txt"

            destination_label = (
                label_destination / label_name
            )

            with open(destination_label, "w") as label_file:
                label_file.write("\n".join(label_lines))

            processed += 1

            if processed % 500 == 0:
                print(
                    f"Processed {processed}/{limit} images..."
                )

    print()
    print(f"Completed {split}: {processed} images")


# ============================================================
# CREATE data.yaml
# ============================================================

def create_data_yaml():

    yaml_content = """path: .
train: dataset/images/train
val: dataset/images/val

names:
  0: pedestrian
  1: rider
  2: car
  3: truck
  4: bus
  5: train
  6: motorcycle
  7: bicycle
  8: traffic light
  9: traffic sign
"""

    yaml_path = PROJECT_ROOT / "dataset" / "data.yaml"

    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print()
    print(f"Created: {yaml_path}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("BDD100K → YOLO DATASET CONVERSION")
    print("=" * 60)

    print()
    print(f"BDD100K root:")
    print(BDD_ROOT)

    print()
    print(f"Project root:")
    print(PROJECT_ROOT)

    # Check BDD100K location

    if not BDD_ROOT.exists():
        print()
        print("ERROR: BDD100K root folder does not exist!")
        print(BDD_ROOT)
        exit(1)

    print()
    print("BDD100K root found successfully.")

    # Process datasets

    process_split("train", TRAIN_LIMIT)
    process_split("val", VAL_LIMIT)

    # Create YAML

    create_data_yaml()

    print()
    print("=" * 60)
    print("DATASET CONVERSION FINISHED")
    print("=" * 60)