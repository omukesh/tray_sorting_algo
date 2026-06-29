# tray_helpers.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import cv2
import asyncio
import numpy as np
from sklearn.cluster import KMeans
from sqlmodel import select
from models import TrayConfig # Import the table class model cleanly

EMPTY_STATE: int = 0
ARUCO_ERROR: int = 4
TRAY_NOT_EMPTY: int = 1
MISSING_INFERENCE: int = 5
MISSING_MARK: int = 0
MIN_CONFIDENCE: float = 0.5

ROI_BOOLEAN: Tuple[int, int, int, int] = (180, 780, 500, 315)
ROI_POINTER: Tuple[int, int, int, int] = (1400, 780, 500, 315)


async def fetch_tray_config_by_id(session: Any, identifier: Union[int, str]) -> Optional[Dict[str, Any]]:
    """
    Unified synchronous lookup layer.
    """
    if session is None or identifier is None:
        return None
        
    def sync_query():
        if isinstance(identifier, int):
            stmt = select(TrayConfig).where(TrayConfig.aruco_id == identifier)
        else:
            stmt = select(TrayConfig).where(TrayConfig.part_number == str(identifier).strip())
            
        # Execute synchronously to prevent ScalarResult await crashes
        result = session.execute(stmt)
        row = result.first()
        
        if row and isinstance(row, tuple):
            row = row[0]
        return row

    try:
        # Offload the synchronous query to a background thread and await its completion safely!
        loop = asyncio.get_running_loop()
        row = await loop.run_in_executor(None, sync_query)

        if row:
            return {
                "aruco_id": row.aruco_id,
                "part_number": row.part_number,
                "class_name": row.class_name,
                "et_rows": row.et_rows,
                "et_cols": row.et_cols,
                "ft_rows": row.ft_rows,
                "ft_cols": row.ft_cols
            }
    except Exception as e:
        print(f"[Critical] Error fetching layout config asynchronously: {str(e)}")
        
    return None
    
def is_inside_roi(center: Tuple[int, int], roi: Tuple[int, int, int, int]) -> bool:
    x, y = center
    rx, ry, rw, rh = roi
    return (rx <= x <= rx + rw) and (ry <= y <= ry + rh)

def is_filling_blade_class(cls_name: str) -> bool:
    return cls_name.startswith("blade_") and cls_name != "blade_generic"

def detect_aruco_ids_and_first_corners(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)

    valid_ids, corners_meta = [], []
    boolean_present, pointer_present = False, False
    pointer_id = 99

    if ids is None:
        return valid_ids, corners_meta, boolean_present, pointer_present, pointer_id

    for i, aid in enumerate(ids.flatten()):
        aid = int(aid)
        pts = corners[i].reshape((4, 2)).astype(int)
        center = (int(pts[:, 0].mean()), int(pts[:, 1].mean()))

        if aid == 0:
            if is_inside_roi(center, ROI_BOOLEAN):
                boolean_present = True
                valid_ids.append(aid)
                corners_meta.append((pts, center))
        elif 1 <= aid <= 31 and aid not in [17, 37]:
            if is_inside_roi(center, ROI_POINTER):
                pointer_present = True
                pointer_id = aid
                valid_ids.append(aid)
                corners_meta.append((pts, center))

    return valid_ids, corners_meta, boolean_present, pointer_present, pointer_id

def decide_layout_and_primary_id(boolean_present, pointer_present, pointer_id):
    if boolean_present and pointer_present:
        return pointer_id, 0, True
    if (not boolean_present) and pointer_present:
        return pointer_id, 1, True
    return 99, 5, False

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

def build_grid_kmeans(objects: List[Dict], rows: int, cols: int) -> Tuple[List[List[Optional[Dict]]], int, int]:
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

def analyze_grid(grid: List[List[Optional[Dict]]], tray_type: int, expected_blade_class: str) -> Tuple[List[List[int]], List[int], bool, List[int]]:
    rows, cols = len(grid), len(grid[0])
    occupancy, blades, missing_elements = [], [], []
    
    # COLUMN-MAJOR REVOLUTION: Put the column loop on the outside!
    for c in range(cols):
        col_data = []
        for rb in range(rows):
            # Maintain your bottom-to-top industrial tracking indexing standard
            rt = rows - 1 - rb
            cell = grid[rt][c]
            slot_id = c * rows + rb + 1
            
            if cell is None:
                col_data.append(MISSING_MARK)
                missing_elements.append(slot_id)
                continue
                
            cls = cell["cls"]
            if tray_type == 0:
                if cls == "slot_empty": col_data.append(0)
                elif cls == "blade_generic":
                    col_data.append(1)
                    blades.append(slot_id)
                else: col_data.append(0)
            else:
                if cls == "slot": col_data.append(0)
                elif cls == expected_blade_class:
                    col_data.append(1)
                    blades.append(slot_id)
                elif is_filling_blade_class(cls):
                    col_data.append(1)
                    blades.append(slot_id)
                elif cls == "missing_inference_void":
                    col_data.append(MISSING_MARK)
                    missing_elements.append(slot_id)
                else: col_data.append(0)
                    
            if cell:
                cell["computed_slot_id"] = slot_id
                
        # Append the entire column array block completely down to the main grid
        occupancy.append(col_data)
        
    return occupancy, blades, len(missing_elements) > 0, sorted(missing_elements)

def compute_tray_bbox_from_masks(objects: List[Dict], H: int, W: int) -> Tuple[int, int, int, int]:
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

def compute_expected_slot_centers_from_bbox(rows: int, cols: int, bbox: Tuple[int, int, int, int]) -> Dict[int, Tuple[int, int]]:
    min_x, min_y, max_x, max_y = bbox
    cell_w, cell_h = (max_x - min_x) / cols, (max_y - min_y) / rows
    centers = {}
    for rb in range(rows):
        rt = rows - 1 - rb
        for c in range(cols):
            slot_id = c * rows + rb + 1
            centers[slot_id] = (int(min_x + (c + 0.5) * cell_w), int(min_y + (rt + 0.5) * cell_h))
    return centers

def render_overlay(img: np.ndarray, grid: List[List[Optional[Dict]]], tray_id: int, centers: Dict[int, Tuple[int, int]], corners_meta: List) -> np.ndarray:
    out = img.copy()
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
    return out

def build_response(occupancy, blades, tray_id, rows, cols, tray_type, status, missing_elements, part_number="PN-UNKNOWN-VOID"):
    return {
        "Part_Number": part_number, "Tray_ID": tray_id, "tray_type": tray_type, "tray_fill_status": status,
        "count": len(blades), "occupancy_grid": occupancy, "blade_elements": sorted(blades),
        "missing_elements": sorted(missing_elements), "rows": rows, "cols": cols,
    }