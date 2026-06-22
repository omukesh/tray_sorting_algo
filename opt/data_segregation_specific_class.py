import os
import json
import random
import shutil
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

# Dataset root (recursive search)
DATASET_ROOT = "/home/mdl/Documents/model_training/Total_trainable/images"

# Output folder
OUTPUT_DIR = "/home/mdl/Videos/mukesh/tray_sorting_algo/tray_random_data"

# Required slot_empty counts
TARGET_TOTALS = [54, 40, 35]

# Random seed
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# ============================================================
# HELPERS
# ============================================================

def find_all_jsons(root_dir):

    json_files = []

    for root, _, files in os.walk(root_dir):

        for file in files:

            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))

    return json_files


def load_json(json_path):

    with open(json_path, "r") as f:
        return json.load(f)


def get_image_path(json_path, json_data):

    # Try using imagePath from json
    image_name = json_data.get("imagePath")

    if image_name:

        candidate = os.path.join(
            os.path.dirname(json_path),
            image_name
        )

        if os.path.exists(candidate):
            return candidate

    # fallback using same filename
    stem = Path(json_path).stem
    parent = Path(json_path).parent

    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:

        candidate = parent / f"{stem}{ext}"

        if candidate.exists():
            return str(candidate)

    return None


def extract_classes(json_data):

    classes = []

    for shape in json_data.get("shapes", []):

        label = shape.get("label")

        if label:
            classes.append(label)

    return classes


def safe_copy(src, dst):

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


# ============================================================
# MAIN
# ============================================================

all_jsons = find_all_jsons(DATASET_ROOT)

print(f"\nFound {len(all_jsons)} json files\n")

# Store matches
matches = {
    54: [],
    40: [],
    35: []
}

# ============================================================
# PROCESS
# ============================================================

for json_path in all_jsons:

    try:
        data = load_json(json_path)

    except Exception as e:
        print(f"Failed reading: {json_path}")
        continue

    image_path = get_image_path(json_path, data)

    if image_path is None:
        continue

    classes = extract_classes(data)

    if len(classes) == 0:
        continue

    unique_classes = set(classes)

    # --------------------------------------------------------
    # ONLY slot_empty allowed
    # --------------------------------------------------------

    if unique_classes != {"slot_empty"}:
        continue

    slot_empty_count = len(classes)

    if slot_empty_count in TARGET_TOTALS:

        matches[slot_empty_count].append(image_path)

# ============================================================
# COPY RANDOM MATCHES
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\nCopying matching images...\n")

for total in TARGET_TOTALS:

    images = matches[total]

    if len(images) == 0:

        print(f"No exclusive slot_empty image found with total={total}")
        continue

    chosen = random.choice(images)

    dst_dir = os.path.join(
        OUTPUT_DIR,
        f"slot_empty_total_{total}"
    )

    os.makedirs(dst_dir, exist_ok=True)

    ext = Path(chosen).suffix

    dst_path = os.path.join(
        dst_dir,
        f"sample{ext}"
    )

    safe_copy(chosen, dst_path)

    print(f"Copied one image with ONLY slot_empty count = {total}")

print("\nDONE.")
