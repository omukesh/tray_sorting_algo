# tray_helpers.py
from __future__ import annotations
import os
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans

# ============================================================
# CONFIG & HARDENED BOUNDS
# ============================================================
SAVE_DIR: str = "./tray"
os.makedirs(SAVE_DIR, exist_ok=True)

EMPTY_STATE: int = 0
VALID_CONTENT: int = 1
ARUCO_ERROR: int = 4
VISION_INCOMPLETE: int = 5

MISSING_MARK: int = 5
MIN_CONFIDENCE: float = 0.5
DB_PATH: str = "tray_config.db"

# Shared ROI Boundaries for ArUco Validation
ROI_BOOLEAN: Tuple[int, int, int, int] = (180, 780, 500, 315)
ROI_POINTER: Tuple[int, int, int, int] = (1400, 780, 500, 315)

# ============================================================
# PERSISTENT LOCAL STORAGE DB INTERFACE
# ============================================================

def fetch_tray_config_by_id(aruco_id: int, db_path: str = DB_PATH) -> Optional[Dict]:
    """Fetches dimensions and part details strictly using local SQLite file."""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT part_number, class_name, et_rows, et_cols, ft_rows, ft_cols 
            FROM tray_configs 
            WHERE aruco_id = ?
        """, (aruco_id,))
        
        row = cursor.fetchone()
        conn.close()
        if row:
            return dict(row)
    except Exception:
        pass 
    return None

# ============================================================
# GEOMETRY HELPERS
# ============================================================

def is_inside_roi(center: Tuple[int, int], roi: Tuple[int, int, int, int]) -> bool:
    x, y = center
    rx, ry, rw, rh = roi
    return (rx <= x <= rx + rw) and (ry <= y <= ry + rh)

def is_filling_blade_class(cls_name: str) -> bool:
    return cls_name.startswith("blade_") and cls_name != "blade_generic"

# ============================================================
# HARDENED ARUCO FILTERING LAYER
# ============================================================

def detect_aruco_ids_and_first_corners(img: np.ndarray):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    corners, ids, _ = aruco_detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        return [], []

    valid_ids, valid_corners_list = [], []
    for i, aruco_id in enumerate(ids.flatten()):
        aid = int(aruco_id)
        pts = corners[i].reshape((4, 2)).astype(int)
        center = (int(pts[:, 0].mean()), int(pts[:, 1].mean()))

        if aid == 0:
            if is_inside_roi(center, ROI_BOOLEAN):
                valid_ids.append(aid)
                valid_corners_list.append((pts, center))
        elif 1 <= aid <= 31 and aid != 17:
            if is_inside_roi(center, ROI_POINTER):
                valid_ids.append(aid)
                valid_corners_list.append((pts, center))

    if not valid_ids:
        return [], []
    return valid_ids, valid_corners_list

def decide_layout_and_primary_id(ids: List[int]) -> Tuple[int, int]:
    if not ids:
        return 99, 5

    ids_set = set(ids)
    if 0 in ids_set:
        tray_type = 0
        pointer_ids = [i for i in ids_set if i != 0]
        tray_id = pointer_ids[0] if pointer_ids else 99
    else:
        tray_type = 1
        tray_id = list(ids_set)[0]

    return tray_id, tray_type

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
        
        objects.append({
            "cls": det["name"],
            "center": ((x1 + x2) // 2, (y1 + y2) // 2),
            "mask": mask,
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
# GRID ANALYSIS AND EVALUATION PASS
# ============================================================

def analyze_grid(grid: List[List[Optional[Dict]]], tray_type: int, expected_blade_class: str):
    rows, cols = len(grid), len(grid[0])
    occupancy, blades, missing_elements = [], [], []
    
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
                if cls == "slot_empty":
                    row_data.append(0)
                elif cls == "blade_generic":
                    row_data.append(1)
                    blades.append(slot_id)
                else:
                    row_data.append(0)
            else:
                if cls == "slot":
                    row_data.append(0)
                elif cls == expected_blade_class:
                    row_data.append(1)
                    blades.append(slot_id)
                elif is_filling_blade_class(cls):
                    row_data.append(1)
                    blades.append(slot_id)
                elif cls == "missing_inference_void":
                    row_data.append(MISSING_MARK)
                    missing_elements.append(slot_id)
                else:
                    row_data.append(0)
                    
            cell["computed_slot_id"] = slot_id
        occupancy.append(row_data)
        
    return occupancy, blades, len(missing_elements) > 0, sorted(missing_elements)

# ============================================================
# BOUNDING AND RENDERING HELPERS
# ============================================================

def compute_tray_bbox_from_masks(objects: List[Dict], H: int, W: int):
    xs, ys = [], []
    for o in objects:
        ys_m, xs_m = np.where(o["mask"] == 1)
        if xs_m.size:
            xs.extend(xs_m.tolist())
            ys.extend(ys_m.tolist())
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

def render_and_save_overlay(img, grid, tray_id, status, centers, save_dir, valid_ids, corners_meta):
    out = img.copy()
    
    # Draw Thin Boundary ROI Boxes
    bx, by, bw, bh = ROI_BOOLEAN
    cv2.rectangle(out, (bx, by), (bx + bw, by + bh), (255, 200, 0), 1)
    px, py, pw, ph = ROI_POINTER
    cv2.rectangle(out, (px, py), (px + pw, py + ph), (0, 255, 255), 1)

    if corners_meta:
        for pts, center in corners_meta:
            top_y = int(np.min(pts[:, 1]))
            top_x = int(pts[np.argmin(pts[:, 1]), 0])
            matched_id = 0 if is_inside_roi(center, ROI_BOOLEAN) else tray_id
            cv2.putText(out, f"ID: {matched_id}", (top_x - 15, top_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.polylines(out, [pts], True, (0, 255, 0), 1)

    rows, cols = len(grid), len(grid[0])
    for r in range(rows):
        for c in range(cols):
            rb = rows - 1 - r
            slot_id = c * rows + rb + 1
            cell = grid[r][c]
            if cell is None:
                cx, cy = centers.get(slot_id, (0, 0))
                cv2.rectangle(out, (cx-30, cy-30), (cx+30, cy+30), (0, 0, 255), 1)
                cv2.putText(out, str(slot_id), (cx-25, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                continue
            
            if "blade" in cell["cls"]:
                contours, _ = cv2.findContours((cell["mask"]*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(out, contours, -1, (0, 165, 255), 1)
                
            cv2.putText(out, str(slot_id), (cell["center"][0]-12, cell["center"][1]+14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    path = os.path.join(save_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_tray{tray_id}_status{status}.jpg")
    cv2.imwrite(path, out)
    return path

def build_response(occupancy, blades, tray_id, rows, cols, tray_type, status, path, missing_elements, part_number="PN-UNKNOWN-VOID"):
    if status == VALID_CONTENT:
        count = len(blades)
    elif status in [EMPTY_STATE, ARUCO_ERROR]:
        count = sum(v == 0 for row in occupancy for v in row)
    elif status == VISION_INCOMPLETE:
        if tray_type == 0:
            count = len(blades)
        else:
            # STATUS 5 BYPASS INVENTORY RULE
            count = max(0, (rows * cols) - sum(v == 0 for row in occupancy for v in row))
    else:
        count = len(blades)

    return {
        "Part_Number": part_number,
        "Tray_ID": tray_id, 
        "tray_type": tray_type, 
        "tray_fill_status": status, 
        "count": count,
        "image_path": path, 
        "occupancy_grid": occupancy, 
        "blade_elements": sorted(blades),
        "missing_elements": missing_elements if status == VISION_INCOMPLETE else [],
        "rows": rows, 
        "cols": cols
    }