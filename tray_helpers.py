# tray_helpers.py
# Pure helper functions — no model, no state, no side effects.
# Preserves the EXACT validated logic from tray_inspect.py (final GPT version).
from __future__ import annotations

import os
import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans

# ============================================================
# CONFIG  (shared — imported by tray_analyzer.py)
# ============================================================
SAVE_DIR: str = "./tray"
os.makedirs(SAVE_DIR, exist_ok=True)

SLOT_CLASS: str  = "slot"
ANGLE_DEG: float = 30.0
PATCH_SIZE: int  = 15
MISSING_MARK: int = 5
MIN_CONF = 0.5

TRAY_LAYOUTS: Dict[int, Tuple[int, int]] = {
    
    1: (5, 8),
    2: (5, 8),
    3: (3, 4),
    4: (3, 4),
    5: (3, 3),
    6: (2, 4),
    7: (3, 4),
    # extend as needed
}

SKU_IDS: set = set(range(1, 31))



# ============================================================
# BASIC GEOMETRY
# ============================================================

def mask_area(mask: np.ndarray) -> int:
    return int(np.sum(mask == 1))

def is_blade_class(cls_name: str):
    return cls_name.startswith("blade") and cls_name != "blade_generic"

def chord_at_angle(
    mask: np.ndarray, angle_deg: float
) -> Tuple[float, Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    """Chord width at angle_deg through mask centroid."""
    ys, xs = np.where(mask == 1)
    if xs.size == 0:
        return 0.0, None, None

    cx, cy = float(xs.mean()), float(ys.mean())
    theta  = math.radians(angle_deg)
    dx, dy = math.cos(theta), math.sin(theta)
    n = math.hypot(dx, dy); dx /= n; dy /= n
    H, W = mask.shape

    def march(sign: int) -> Tuple[int, int]:
        t: float = 0.0
        last = (int(cx), int(cy))
        while True:
            x = int(round(cx + sign * t * dx))
            y = int(round(cy - sign * t * dy))
            if x < 0 or x >= W or y < 0 or y >= H or mask[y, x] == 0:
                return last
            last = (x, y); t += 1.0

    p1, p2 = march(+1), march(-1)
    return round(math.hypot(p1[0]-p2[0], p1[1]-p2[1]), 2), p1, p2


# ============================================================
# ARUCO + COLOR
# ============================================================


def infer_empty_layout(objs):
    """
    Infer empty tray layout based on slot_empty detections
    """

    total_slots = sum(
        1 for o in objs if ("slot" in o["cls"] or "blade" in o["cls"])
    )

    if total_slots >= 50:
        return (6, 9)
    elif total_slots >= 30:
        return (5, 7)
    elif total_slots >= 18:
        return (4, 5)
    else:
        return (3, 3)


def infer_filled_layout(tray_id, TRAY_LAYOUTS):
    """
    Get layout from config for filled trays
    """
    return TRAY_LAYOUTS.get(tray_id, (5, 8))


def detect_aruco_ids_and_first_corners(
    img: np.ndarray,
) -> Tuple[List[int], Optional[np.ndarray]]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ar = cv2.aruco
    try:
        dic = ar.getPredefinedDictionary(ar.DICT_4X4_250)
        det = ar.ArucoDetector(dic, ar.DetectorParameters())
        corners, ids, _ = det.detectMarkers(gray)
    except Exception:
        dic = ar.getPredefinedDictionary(ar.DICT_4X4_250)
        corners, ids, _ = ar.detectMarkers(gray, dic)  # type: ignore

    if ids is None or len(ids) == 0:
        return [], None
    uniq  = sorted(set(int(i) for i in ids.flatten()))
    first = corners[0].reshape(-1, 2).astype(int)
    return uniq, first


# def decide_layout_and_primary_id(
#     detected_ids: List[int],
# ) -> Tuple[int, int, int, int]:
#     """
#     Returns (rows, cols, tray_id, tray_type).
#     tray_type: 0=empty-tray  1=filled-tray  5=aruco-error
#     """
#     ids_set = set(detected_ids)

#     if 0 in ids_set:
#         skus = [i for i in ids_set if i in SKU_IDS and i != 0]
#         if skus:
#             sku = skus[0]
#             rows, cols = TRAY_LAYOUTS.get(sku, TRAY_LAYOUTS[0])
#             return rows, cols, sku, 0

#     for i in detected_ids:
#         if i in TRAY_LAYOUTS:
#             rows, cols = TRAY_LAYOUTS[i]
#             return rows, cols, i, 1

#     rows, cols = TRAY_LAYOUTS[0]
#     return rows, cols, 0, 5

def decide_layout_and_primary_id(ids):
    """
    Decide tray type and primary ID

    Returns:
        tray_id, tray_type
    """

    if not ids:
        return -1, -1  # no aruco

    ids_set = set(ids)

    # EMPTY tray (has ID 0)
    if 0 in ids_set:
        tray_type = 0  # EMPTY
        # pointer ID = any non-zero
        pointer_ids = [i for i in ids_set if i != 0]
        tray_id = pointer_ids[0] if pointer_ids else -1

    else:
        tray_type = 1  # FILLING
        tray_id = list(ids_set)[0]

    return tray_id, tray_type


def draw_aruco_ids(img : np.ndarray, color=(255, 255, 255)):
    """
    Draw detected ArUco IDs on image
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(dictionary)

    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:
        for i, corner in enumerate(corners):
            pts = corner.reshape((4, 2)).astype(int)
            cx = int(pts[:, 0].mean())
            cy = int(pts[:, 1].mean())

            cv2.polylines(img, [pts], True, color, 2)

            cv2.putText(
                img,
                str(ids[i][0]),
                (cx - 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

    return img

# ============================================================
# YOLO OBJECT EXTRACTION
# ============================================================

def extract_objects_from_result(detections, h, w):
    objs = []

    for det in detections:
        cls_name = det["name"]
        conf = det["confidence"]

        if conf < MIN_CONF:
            continue

        x1, y1, x2, y2 = map(int, det["box"])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if det["mask_array"] is not None:
            mask = det["mask_array"].astype("uint8")
        else:
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[y1:y2, x1:x2] = 1

        # ✅ width ONLY for bladeXXX
        width = 0.0
        pA, pB = None, None

        if is_blade_class(cls_name):
            width, pA, pB = chord_at_angle(mask, ANGLE_DEG)

        objs.append({
            "cls": cls_name,
            "center": (cx, cy),
            "mask": mask,
            "bbox": (x1, y1, x2, y2),
            "width": width,
            "pA": pA,
            "pB": pB,
            "confidence": conf
        })

    return objs


# ============================================================
# GRID ASSIGNMENT  (KMeans — stable under missing detections)
# ============================================================

def build_grid_kmeans(objs, rows, cols):
    """
    Robust grid builder:
    - KMeans for structure
    - multiple candidates per cell
    - selects best match
    """

    if not objs:
        return [[None]*cols for _ in range(rows)], rows, cols

    centers = np.array([o["center"] for o in objs], dtype=np.float32)

    y = centers[:, 1].reshape(-1, 1)
    x = centers[:, 0].reshape(-1, 1)

    row_km = KMeans(n_clusters=min(rows, len(y)), random_state=0, n_init=10).fit(y)
    col_km = KMeans(n_clusters=min(cols, len(x)), random_state=0, n_init=10).fit(x)

    row_centers = row_km.cluster_centers_.flatten()
    col_centers = col_km.cluster_centers_.flatten()

    row_order = np.argsort(row_centers)
    col_order = np.argsort(col_centers)

    row_map = {old: new for new, old in enumerate(row_order)}
    col_map = {old: new for new, old in enumerate(col_order)}

    # 🔥 STEP 1: collect ALL candidates per cell
    temp_grid = [[[] for _ in range(cols)] for _ in range(rows)]

    for o, r, c in zip(objs, row_km.labels_, col_km.labels_):
        rt = row_map[r]
        cc = col_map[c]

        # distance to cluster center
        dist = abs(o["center"][1] - row_centers[r]) + abs(o["center"][0] - col_centers[c])

        temp_grid[rt][cc].append((o, dist))

    # 🔥 STEP 2: pick BEST candidate per cell
    grid = [[None for _ in range(cols)] for _ in range(rows)]

    for r in range(rows):
        for c in range(cols):
            if temp_grid[r][c]:
                best_obj = min(temp_grid[r][c], key=lambda x: x[1])[0]
                grid[r][c] = best_obj

    # 🔥 STEP 3: assign slot IDs (bottom-left origin)
    for r in range(rows):
        rb = rows - 1 - r  # bottom-left numbering
        for c in range(cols):
            if grid[r][c] is not None:
                grid[r][c]["row"] = rb
                grid[r][c]["col"] = c
                grid[r][c]["slot_id"] = rb * cols + c + 1

    return grid, rows, cols

# ============================================================
# EXPECTED SLOT CENTRES  (for missing-inference rendering)
# ============================================================

def compute_tray_bbox_from_masks(
    objs: List[Dict], H: int, W: int, margin_ratio: float = 0.05
) -> Tuple[int, int, int, int]:
    """Tray interior bbox from detected masks, shrunk inward to avoid walls/ArUco."""
    xs: List[int] = []
    ys: List[int] = []
    for o in objs:
        ys_m, xs_m = np.where(o["mask"] == 1)
        if xs_m.size:
            xs.extend(xs_m.tolist())
            ys.extend(ys_m.tolist())
    if not xs:
        return (0, 0, W, H)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    dx = int((max_x - min_x) * margin_ratio)
    dy = int((max_y - min_y) * margin_ratio)
    return (min_x+dx, min_y+dy, max_x-dx, max_y-dy)


def compute_expected_slot_centers_from_bbox(
    rows: int, cols: int, tray_bbox: Tuple[int, int, int, int]
) -> Dict[int, Tuple[int, int]]:
    """Physical slot centres on a uniform grid inside tray_bbox (bottom-left numbering)."""
    min_x, min_y, max_x, max_y = tray_bbox
    cell_w = (max_x - min_x) / cols
    cell_h = (max_y - min_y) / rows
    centers: Dict[int, Tuple[int, int]] = {}
    for rb in range(rows):
        rt = rows - 1 - rb
        for c in range(cols):
            cx      = int(min_x + (c  + 0.5) * cell_w)
            cy      = int(min_y + (rt + 0.5) * cell_h)
            slot_id = rb * cols + c + 1
            centers[slot_id] = (cx, cy)
    return centers


# ============================================================
# GRID ANALYSIS
# ============================================================

def analyze_grid(
    grid: List[List[Optional[Dict]]],
) -> Tuple[List[List[int]], List[int], List[float], bool]:
    """
    Returns (occupancy_grid, blade_slot_ids, blade_widths, missing_flag).
    Rows iterated bottom-to-top to match slot numbering.
    """
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    occ:    List[List[int]]  = []
    blades: List[int]        = []
    widths: List[float]      = []
    missing = False

    for rb in range(rows):
        rt  = rows - 1 - rb
        row: List[int] = []
        for c in range(cols):
            cell = grid[rt][c]
            if cell is None:
                row.append(MISSING_MARK)
                missing = True

            elif "slot" in cell["cls"]:
                row.append(0)

            elif "blade" in cell["cls"]:
                row.append(1)
                blades.append(cell["slot_id"])

                if is_blade_class(cell["cls"]):
                    widths.append(cell["width"])
        occ.append(row)

    return occ, blades, widths, missing


# ============================================================
# RENDER & SAVE
# ============================================================

def render_and_save_overlay(
    img: np.ndarray,
    grid: List[List[Optional[Dict]]],
    tray_id: int,
    status: int,
    expected_centers: Dict[int, Tuple[int, int]],
    save_dir: str = SAVE_DIR,
) -> str:
    """
    Clean operator-friendly overlay:
      🟢 Slot → number only
      🟠 Blade → contour + number + width
      🔴 Missing → box + number
      🟠 ArUco → orange
    """

    out = img.copy()

    # --- Draw ArUco IDs in ORANGE ---
    out = draw_aruco_ids(out, color=(0, 165, 255))  # orange

    rows = len(grid)
    cols = len(grid[0]) if rows else 0

    for r in range(rows):
        for c in range(cols):

            slot_id = (rows - 1 - r) * cols + c + 1
            cell = grid[r][c]

            # =========================
            # 🔴 MISSING
            # =========================
            if cell is None:
                cx, cy = expected_centers[slot_id]

                w_box, h_box = 60, 60
                x1, y1 = int(cx - w_box / 2), int(cy - h_box / 2)
                x2, y2 = int(cx + w_box / 2), int(cy + h_box / 2)

                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)

                cv2.putText(
                    out, str(slot_id), (x1 + 5, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                )
                continue

            cls = cell["cls"]

            # =========================
            # 🟠 BLADE → contour only
            # =========================
            if "blade" in cls:

                mask_u8 = (cell["mask"] * 255).astype(np.uint8)
                cnts, _ = cv2.findContours(
                    mask_u8,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                if cnts:
                    cv2.drawContours(out, cnts, -1, (0, 165, 255), 2)  # orange

                # width line (only real blades)
                if (
                    cls != "blade_generic"
                    and cell.get("pA") is not None
                    and cell.get("pB") is not None
                ):
                    cv2.line(out, cell["pA"], cell["pB"], (0, 255, 255), 2)

                    mid = (
                        (cell["pA"][0] + cell["pB"][0]) // 2,
                        (cell["pA"][1] + cell["pB"][1]) // 2,
                    )

                    cv2.putText(
                        out,
                        f"{cell['width']:.1f}px",
                        (mid[0] + 6, mid[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        2,
                    )

            # =========================
            # 🟢 NUMBER (for all detected)
            # =========================
            ys_m, xs_m = np.where(cell["mask"] == 1)
            if xs_m.size:
                cx, cy = int(xs_m.mean()), int(ys_m.mean())
            else:
                cx, cy = cell["center"]

            cv2.putText(
                out, str(slot_id), (cx - 12, cy + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

    fname = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_tray{tray_id}_status{status}.jpg"
    path = os.path.join(save_dir, fname)

    cv2.imwrite(path, out)
    return path


# ============================================================
# RESPONSE BUILDER
# ============================================================

def build_response(
    occ:       List[List[int]],
    blades:    List[int],
    widths:    List[float],
    tray_id:   int,
    rows:      int,
    cols:      int,
    tray_type: int,
    status:    int,
    img_path:  str,
) -> Dict:
    blade_count = len(blades)
    slot_count  = sum(v == 0 for row in occ for v in row)
    if status == 1:
        count = blade_count
    elif status in (0, 4):
        count = slot_count
    else:
        count = blade_count if blade_count else slot_count

    return {
        "Tray_ID":          tray_id,
        "tray_type":        tray_type,
        "tray_fill_status": status,
        "count":            count,
        "image_path":       img_path,
        "occupancy_grid":   occ,
        "blade_elements":   blades,
        "top_view_widths":  widths,
        "rows":             rows,
        "cols":             cols,
    }