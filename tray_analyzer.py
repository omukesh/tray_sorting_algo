# tray_analyzer.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np

from tray_helpers import (
    MISSING_INFERENCE, analyze_grid, build_grid_kmeans, build_response, 
    detect_aruco_ids_and_first_corners, decide_layout_and_primary_id, 
    extract_objects_from_result, compute_tray_bbox_from_masks, 
    compute_expected_slot_centers_from_bbox, render_overlay,
    fetch_tray_config_by_id, EMPTY_STATE, TRAY_NOT_EMPTY, ARUCO_ERROR,
)

class TrayAnalyzer:
    def __init__(self, client, camera=None):
        self.client = client
        self.camera = camera

    async def analyze_frame(self, session, frame: np.ndarray, detections: List[Dict]) -> Tuple[dict, np.ndarray]:
        if frame is None or frame.size == 0:
            raise ValueError("cannot analyse on unallocated frame buffer.")

        H, W = frame.shape[:2]
        (_, corners_meta, boolean_present, pointer_present, pointer_id) = detect_aruco_ids_and_first_corners(frame)
        (tray_id, tray_type, aruco_valid) = decide_layout_and_primary_id(boolean_present, pointer_present, pointer_id)

        detected_objects = extract_objects_from_result(detections, H, W)
        aruco_error = False

        if not aruco_valid:
            aruco_error = True
            is_physically_empty = (
                len(detected_objects) > 0 and all(obj["cls"] in ["slot_empty", "blade_generic"] for obj in detected_objects)
            )
            tray_type = 0 if is_physically_empty else 1
            tray_id = 99

        config = None
        if pointer_present:
            config = await fetch_tray_config_by_id(session, pointer_id)

        if config:
            rows = config["et_rows"] if tray_type == 0 else config["ft_rows"]
            cols = config["et_cols"] if tray_type == 0 else config["ft_cols"]
            part_number = config["part_number"]
            expected_class = config["class_name"]
        else:
            rows, cols = (6, 9) if tray_type == 0 else (5, 8)
            part_number, expected_class = "UNKNOWN", "blade_generic"

        print(f"[INFO] Mode: {'EMPTY' if tray_type == 0 else 'FILLING'} | Matrix Locked: {rows}x{cols}")
        
        grid, rows, cols = build_grid_kmeans(detected_objects, rows, cols)
        (occupancy, blades, has_missing, missing_slots) = analyze_grid(grid, tray_type, expected_class)

        status = ARUCO_ERROR if aruco_error else (MISSING_INFERENCE if has_missing else (TRAY_NOT_EMPTY if blades else EMPTY_STATE))

        bbox = compute_tray_bbox_from_masks(detected_objects, H, W)
        centers = compute_expected_slot_centers_from_bbox(rows, cols, bbox)
        overlay_canvas = render_overlay(img=frame, grid=grid, tray_id=tray_id, centers=centers, corners_meta=corners_meta)

        response_data = build_response(
            occupancy=occupancy, blades=blades, tray_id=tray_id, rows=rows, cols=cols,
            tray_type=tray_type, status=status, missing_elements=missing_slots, part_number=part_number
        )
        return response_data, overlay_canvas
