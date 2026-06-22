# tray_helpers.py
from __future__ import annotations
import os
import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import sqlite3
import cv2
import numpy as np
from sklearn.cluster import KMeans

# ============================================================
# CONFIG
# ============================================================
SAVE_DIR: str = "./tray"
os.makedirs(SAVE_DIR, exist_ok=True)

MISSING_MARK: int = 5
MIN_CONFIDENCE: float = 0.5

# Shared ROI Boundaries for ArUco Validation
ROI_BOOLEAN: Tuple[int, int, int, int] = (180, 780, 500, 315)
ROI_POINTER: Tuple[int, int, int, int] = (1400, 780, 500, 315)

# Comprehensive Filling Tray Map: Pointer ID -> (rows, cols, expected_blade_class)
TRAY_LAYOUTS_FILLING: Dict[int, Tuple[int, int, str]] = {
    1: (5, 8, "blade35046"),
    2: (5, 8, "blade042"),
    3: (5, 8, "blade052"),
    4: (3, 4, "blade012"),
    5: (3, 3, "blade022"),
    6: (5, 8, "blade072"),
    7: (5, 8, "blade22001"),
    8: (5, 8, "blade22002"),
    9: (5, 8, "blade35032"),
}

# Comprehensive Empty Tray Map: Pointer ID -> (rows, cols)
TRAY_LAYOUTS_EMPTY: Dict[int, Tuple[int, int]] = {
    1: (6, 9),
    2: (6, 9),
    3: (5, 7),
    4: (4, 7),
}

# ============================================================
# GEOMETRY & ROTATIONAL WIDTH (MAX CHORD SWEEP)
# ============================================================

def is_inside_roi(center: Tuple[int, int], roi: Tuple[int, int, int, int]) -> bool:
    x, y = center
    rx, ry, rw, rh = roi
    return (rx <= x <= rx + rw) and (ry <= y <= ry + rh)

def is_filling_blade_class(cls_name: str) -> bool:
    """Returns true if the class belongs to specific filling blades (#3 to #11)."""
    filling_blades = {
        "blade012", "blade022", "blade042", "blade052", 
        "blade072", "blade22001", "blade22002", "blade35032", "blade35046"
    }
    return cls_name in filling_blades

def find_max_chord(mask: np.ndarray) -> Tuple[float, Tuple[int, int], Tuple[int, int]]:
    """
    Sweeps angles to find the absolute maximum width (chord) of the mask.
    Replaces static degree configurations with true geometric maxima.
    """
    ys, xs = np.where(mask == 1)
    if xs.size == 0:
        return 0.0, (0, 0), (0, 0)

    cx, cy = float(xs.mean()), float(ys.mean())
    H, W = mask.shape
    max_width, best_p1, best_p2 = 0.0, (0, 0), (0, 0)

    for degree in range(0, 180, 10):
        theta = math.radians(degree)
        dx, dy = math.cos(theta), math.sin(theta)

        def march(sign: int) -> Tuple[int, int]:
            t = 0.0
            last = (int(cx), int(cy))
            while True:
                nx, ny = int(cx + sign * t * dx), int(cy - sign * t * dy)
                if nx < 0 or nx >= W or ny < 0 or ny >= H or mask[ny, nx] == 0:
                    return last
                last, t = (nx, ny), t + 1.0

        p1, p2 = march(1), march(-1)
        current_w = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
        if current_w > max_width:
            max_width, best_p1, best_p2 = current_w, p1, p2

    return round(max_width, 2), best_p1, best_p2

# ============================================================
# ARUCO HANDLING
# ============================================================

def detect_aruco_ids_and_first_corners(img: np.ndarray):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    corners, ids, _ = aruco_detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        return [], None

    valid_ids, valid_corners = [], []
    for i, aruco_id in enumerate(ids.flatten()):
        if int(aruco_id) == 17:  # hard negated
            continue
        
        pts = corners[i].reshape((4, 2)).astype(int)
        center = (int(pts[:, 0].mean()), int(pts[:, 1].mean()))

        if (int(aruco_id) == 0 and is_inside_roi(center, ROI_BOOLEAN)) or \
           (int(aruco_id) != 0 and is_inside_roi(center, ROI_POINTER)):
            valid_ids.append(int(aruco_id))
            valid_corners.append(corners[i])

    if not valid_ids:
        return [], None
    return sorted(list(set(valid_ids))), valid_corners[0].reshape(-1, 2).astype(int)

def decide_layout_and_primary_id(ids: List[int]) -> Tuple[int, int]:
    if not ids:
        return 99, 5  # No ArUco state

    ids_set = set(ids)
    if 0 in ids_set:
        tray_type = 0
        pointer_ids = [i for i in ids_set if i != 0]
        tray_id = pointer_ids[0] if pointer_ids else 99
    else:
        tray_type = 1
        tray_id = list(ids_set)[0]

    return tray_id, tray_type

def infer_empty_layout_dynamic(objects_count: int) -> Tuple[int, int]:
    if objects_count >= 46: return (6, 9)
    elif 41 <= objects_count <= 45: return (5, 9)
    elif 36 <= objects_count <= 40: return (5, 8)
    elif 30 <= objects_count <= 35: return (5, 7)
    return (3, 3)

# ============================================================
# PROCESSING & EXTRACTION
# ============================================================

def extract_objects_from_result(detections: List[Dict], h: int, w: int) -> List[Dict]:
    objects = []
    for det in detections:
        if det["confidence"] < MIN_CONFIDENCE:
            continue
        x1, y1, x2, y2 = map(int, det["box"])
        mask = det["mask_array"] if det.get("mask_array") is not None else np.zeros((h, w), dtype=np.uint8)
        
        width, pA, pB = 0.0, None, None
        if is_filling_blade_class(det["name"]) or det["name"] == "blade_generic":
            width, pA, pB = find_max_chord(mask)

        objects.append({
            "cls": det["name"],
            "center": ((x1 + x2) // 2, (y1 + y2) // 2),
            "mask": mask,
            "width": width,
            "pA": pA,
            "pB": pB,
            "confidence": det["confidence"]
        })
    return objects

def build_grid_kmeans(objects: List[Dict], rows: int, cols: int):
    if not objects:
        return [[None] * cols for _ in range(rows)], rows, cols

    points = np.array([o["center"] for o in objects], dtype=np.float32)
    row_km = KMeans(n_clusters=min(rows, len(points)), n_init=10, random_state=42).fit(points[:, 1].reshape(-1, 1))
    col_km = KMeans(n_clusters=min(cols, len(points)), n_init=10, random_state=42).fit(points[:, 0].reshape(-1, 1))
    
    row_map = {old: new for new, old in enumerate(np.argsort(row_km.cluster_centers_.flatten()))}
    col_map = {old: new for new, old in enumerate(np.argsort(col_km.cluster_centers_.flatten()))}

    grid = [[None for _ in range(cols)] for _ in range(rows)]
    for o, row_label, col_label in zip(objects, row_km.labels_, col_km.labels_):
        grid[row_map[row_label]][col_map[col_label]] = o

    for r in range(rows):
        for c in range(cols):
            if grid[r][c]:
                grid[r][c]["slot_id"] = c * rows + (rows - 1 - r) + 1
    return grid, rows, cols

# ============================================================
# ANALYSIS WITH EXCLUSIVE CLASS MAPPINGS
# ============================================================

def analyze_grid(grid: List[List[Optional[Dict]]], tray_type: int, expected_blade_class: str):
    rows, cols = len(grid), len(grid[0])
    occupancy, blades, widths, missing_elements = [], [], [], []
    
    for rb in range(rows):
        rt = rows - 1 - rb
        row_data = []
        for c in range(cols):
            cell = grid[rt][c]
            slot_id = c * rows + rb + 1
            
            if cell is None:
                row_data.append(MISSING_MARK)
                missing_elements.append(slot_id)
                continue
                
            cls = cell["cls"]

            if tray_type == 0:
                # --- EMPTY TRAY CLAUSE: Strict Enforcement of 0 & 1 Only ---
                if cls == "slot_empty":
                    row_data.append(0)
                elif cls == "blade_generic":
                    row_data.append(1)
                    blades.append(slot_id)
                    widths.append(cell["width"])
                else:
                    # Mismatched/Unwanted classes inside an empty tray marked as slot_empty fallback
                    row_data.append(0)
            else:
                # --- FILLING TRAY CLAUSE: Strict Enforcement of 2 & expected_blade_class ---
                if cls == "slot":
                    row_data.append(0)
                elif cls == expected_blade_class:
                    row_data.append(1)
                    blades.append(slot_id)
                    widths.append(cell["width"])
                elif is_filling_blade_class(cls):
                    # WRONG SKU Placement matching clause
                    row_data.append(1)
                    blades.append(slot_id)
                    widths.append(cell["width"])
                    cell["wrong_sku_triggered"] = True
                else:
                    row_data.append(0)
                    
            cell["computed_slot_id"] = slot_id
        occupancy.append(row_data)
        
    return occupancy, blades, widths, len(missing_elements) > 0, sorted(missing_elements)

# ============================================================
# RENDER & EXPORTS
# ============================================================

def compute_tray_bbox_from_masks(objects: List[Dict], H: int, W: int):
    xs, ys = [], []
    for o in objects:
        ys_m, xs_m = np.where(o["mask"] == 1)
        if xs_m.size:
            xs.extend(xs_m.tolist()); ys.extend(ys_m.tolist())
    if not xs: return (0, 0, W, H)
    min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
    dx, dy = int((max_x - min_x) * 0.05), int((max_y - min_y) * 0.05)
    return (min_x + dx, min_y + dy, max_x - dx, max_y - dy)

def compute_expected_slot_centers_from_bbox(rows: int, cols: int, bbox: Tuple[int, int, int, int]):
    min_x, min_y, max_x, max_y = bbox
    cell_w, cell_h = (max_x - min_x) / cols, (max_y - min_y) / rows
    centers = {}
    for rb in range(rows):
        rt = rows - 1 - rb
        for c in range(cols):
            slot_id = c * rows + rb + 1
            centers[slot_id] = (int(min_x + (c + 0.5) * cell_w), int(min_y + (rt + 0.5) * cell_h))
    return centers

def render_and_save_overlay(img, grid, tray_id, status, centers, save_dir, valid_ids, corners):
    out = img.copy()
    if valid_ids:
        for aid in valid_ids:
            label = "Boolean ID: 0" if aid == 0 else f"Pointer ID: {aid}"
            cv2.putText(out, label, (50, 50 if aid == 0 else 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 183, 197), 1)

    rows, cols = len(grid), len(grid[0])
    for r in range(rows):
        for c in range(cols):
            rb = rows - 1 - r; slot_id = c * rows + rb + 1; cell = grid[r][c]
            if cell is None:
                cx, cy = centers.get(slot_id, (0, 0))
                cv2.rectangle(out, (cx-30, cy-30), (cx+30, cy+30), (0, 0, 255), 2)
                cv2.putText(out, str(slot_id), (cx-25, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                continue
            
            # Use specific color scheme based on accurate conditions
            if "blade" in cell["cls"]:
                contour_color = (255, 0, 255) if cell.get("wrong_sku_triggered") else (0, 165, 255)
                contours, _ = cv2.findContours((cell["mask"]*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(out, contours, -1, contour_color, 1)
                
                if cell["pA"] and cell["pB"] and not cell.get("wrong_sku_triggered"):
                    cv2.line(out, cell["pA"], cell["pB"], (0, 255, 255), 1)
                    cv2.putText(out, f"{cell['width']:.1f}px", cell["pA"], cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    
            cv2.putText(out, str(slot_id), (cell["center"][0]-12, cell["center"][1]+14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    path = os.path.join(save_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_tray{tray_id}_status{status}.jpg")
    cv2.imwrite(path, out)
    return path

def build_response(occupancy, blades, widths, tray_id, rows, cols, tray_type, status, path, missing_elements):
    count = len(blades) if status == 1 else sum(v == 0 for row in occupancy for v in row)
    return {
        "Tray_ID": tray_id, "tray_type": tray_type, "tray_fill_status": status, "count": count,
        "image_path": path, "occupancy_grid": occupancy, "blade_elements": sorted(blades),
        "missing_elements": missing_elements if status == 5 else [], "top_view_widths": widths,
        "rows": rows, "cols": cols
    }