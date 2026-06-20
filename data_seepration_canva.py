import os
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

# Dataset root (recursive search)
DATASET_ROOT = "/home/mdl/Documents/model_training/Total_trainable/images"

# Output folder
OUTPUT_DIR = "/home/mdl/Videos/mukesh/tray_sorting_algo/tray_random_data"

# Number of random images per class
SAMPLES_PER_CLASS = 3

# Special object totals
SPECIAL_TOTALS = [54, 40, 35]

# Random seed
RANDOM_SEED = 42

# ============================================================
# CLASSES
# ============================================================

CLASS_NAMES = [
    "slot_empty",
    "blade_generic",
    "slot",
    "blade012",
    "blade022",
    "blade042",
    "blade052",
    "blade072",
    "blade22001",
    "blade22002",
    "blade35032",
    "blade35046",
]

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

    # Try imagePath first
    image_name = json_data.get("imagePath")

    if image_name:

        candidate = os.path.join(
            os.path.dirname(json_path),
            image_name
        )

        if os.path.exists(candidate):
            return candidate

    # Fallback: same filename
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

# ------------------------------------------------------------
# STORAGE
# ------------------------------------------------------------

class_samples = defaultdict(list)

only_slot_empty = []
slot_empty_blade_generic = []

special_totals = defaultdict(list)

# ============================================================
# PROCESS ALL JSONS
# ============================================================

for json_path in all_jsons:

    try:
        data = load_json(json_path)

    except Exception as e:
        print(f"Failed reading: {json_path}")
        print(e)
        continue

    image_path = get_image_path(json_path, data)

    if image_path is None:
        print(f"Image missing for: {json_path}")
        continue

    classes = extract_classes(data)

    if len(classes) == 0:
        continue

    unique_classes = set(classes)

    # --------------------------------------------------------
    # PER CLASS STORAGE
    # --------------------------------------------------------

    for cls in unique_classes:

        if cls in CLASS_NAMES:
            class_samples[cls].append(image_path)

    # --------------------------------------------------------
    # SPECIAL CONDITIONS
    # --------------------------------------------------------

    total_objects = len(classes)

    # ONLY slot_empty
    if unique_classes == {"slot_empty"}:
        only_slot_empty.append((image_path, total_objects))

    # slot_empty + blade_generic only
    if unique_classes == {"slot_empty", "blade_generic"}:
        slot_empty_blade_generic.append(
            (image_path, total_objects)
        )

    # TOTAL COUNTS
    if total_objects in SPECIAL_TOTALS:
        special_totals[total_objects].append(image_path)

# ============================================================
# OUTPUT
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# 1. RANDOM 3 IMAGES PER CLASS
# ------------------------------------------------------------

print("Copying per-class samples...\n")

for cls_name in CLASS_NAMES:

    images = list(set(class_samples[cls_name]))

    if len(images) == 0:
        print(f"{cls_name}: No images found")
        continue

    chosen = random.sample(
        images,
        min(SAMPLES_PER_CLASS, len(images))
    )

    cls_dir = os.path.join(
        OUTPUT_DIR,
        "per_class",
        cls_name
    )

    os.makedirs(cls_dir, exist_ok=True)

    for idx, img_path in enumerate(chosen):

        ext = Path(img_path).suffix

        dst = os.path.join(
            cls_dir,
            f"{idx+1}{ext}"
        )

        safe_copy(img_path, dst)

    print(f"{cls_name}: {len(chosen)} copied")

# ------------------------------------------------------------
# 2. ONLY slot_empty
# ------------------------------------------------------------

special_dir = os.path.join(
    OUTPUT_DIR,
    "special_cases"
)

os.makedirs(special_dir, exist_ok=True)

print("\nSelecting ONLY slot_empty image...")

if len(only_slot_empty) > 0:

    img_path, total = random.choice(only_slot_empty)

    ext = Path(img_path).suffix

    dst = os.path.join(
        special_dir,
        f"only_slot_empty_total_{total}{ext}"
    )

    safe_copy(img_path, dst)

    print("Copied ONLY slot_empty image")

else:
    print("No ONLY slot_empty image found")

# ------------------------------------------------------------
# 3. slot_empty + blade_generic
# ------------------------------------------------------------

print("\nSelecting slot_empty + blade_generic image...")

if len(slot_empty_blade_generic) > 0:

    img_path, total = random.choice(
        slot_empty_blade_generic
    )

    ext = Path(img_path).suffix

    dst = os.path.join(
        special_dir,
        f"slot_empty_blade_generic_total_{total}{ext}"
    )

    safe_copy(img_path, dst)

    print("Copied slot_empty + blade_generic image")

else:
    print("No slot_empty + blade_generic image found")

# ------------------------------------------------------------
# 4. TOTALS = 54 / 40 / 35
# ------------------------------------------------------------

print("\nSelecting totals 54 / 40 / 35...\n")

for total in SPECIAL_TOTALS:

    matches = special_totals[total]

    if len(matches) == 0:
        print(f"No image found with total={total}")
        continue

    img_path = random.choice(matches)

    dst_dir = os.path.join(
        OUTPUT_DIR,
        "special_totals",
        f"total_{total}"
    )

    os.makedirs(dst_dir, exist_ok=True)

    ext = Path(img_path).suffix

    dst = os.path.join(
        dst_dir,
        f"sample{ext}"
    )

    safe_copy(img_path, dst)

    print(f"Copied total={total}")

print("\nDONE.")
