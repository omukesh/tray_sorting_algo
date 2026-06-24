# tray_analyzer.py
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

from tray_helpers import (
    SAVE_DIR, 
    analyze_grid, 
    build_grid_kmeans, 
    build_response, 
    detect_aruco_ids_and_first_corners,
    decide_layout_and_primary_id, 
    extract_objects_from_result,
    compute_tray_bbox_from_masks, 
    compute_expected_slot_centers_from_bbox, 
    render_and_save_overlay,
    fetch_tray_config_by_id  
)

class TrayAnalyzer:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def analyze_image(self, image_path: str) -> dict:
        img = cv2.imread(image_path)
        if img is None: 
            raise FileNotFoundError(f"Cannot read image file: {image_path}")
        H, W = img.shape[:2]

        valid_ids, corners_meta = detect_aruco_ids_and_first_corners(img)
        tray_id, tray_type = decide_layout_and_primary_id(valid_ids)
        
        results = self.model(img, verbose=False)[0]
        
        raw_boxes = []
        raw_confidences = []
        raw_class_ids = []
        raw_masks = []

        for i, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            raw_boxes.append([x1, y1, x2 - x1, y2 - y1])
            raw_confidences.append(float(box.conf[0]))
            raw_class_ids.append(int(box.cls[0]))
            if results.masks is not None:
                raw_masks.append(results.masks.data[i].cpu().numpy())
            else:
                raw_masks.append(None)

        indices = cv2.dnn.NMSBoxes(raw_boxes, raw_confidences, score_threshold=0.25, nms_threshold=0.65)
        detections = []
        
        db_config = fetch_tray_config_by_id(tray_id)
        
        if db_config:
            expected_blade_class = db_config["class_name"]
            resolved_part_number = db_config["part_number"]
        else:
            resolved_part_number = "UNKNOWN"
            expected_blade_class = "missing_inference_void" if tray_type == 1 else "blade_generic"

        if len(indices) > 0:
            for idx in indices.flatten():
                x, y, w, h = raw_boxes[idx]
                conf = raw_confidences[idx]
                cls_id = raw_class_ids[idx]
                mask_raw = raw_masks[idx]
                
                raw_cls_name = results.names[cls_id]
                
                if cls_id in range(3, 22):
                    final_name = expected_blade_class if tray_type == 1 else "blade_generic"
                else:
                    final_name = raw_cls_name

                det = {
                    "box": [x, y, x + w, y + h],
                    "name": final_name, 
                    "confidence": conf
                }
                
                if mask_raw is not None:
                    mask_resized = cv2.resize(mask_raw, (W, H), interpolation=cv2.INTER_NEAREST)
                    det["mask_array"] = (mask_resized > 0.5).astype("uint8")
                else:
                    det["mask_array"] = None
                    
                detections.append(det)

        objects = extract_objects_from_result(detections, H, W)
        is_physically_empty = all(o["cls"] in ["slot_empty", "blade_generic"] for o in objects)
        
        if tray_type == 5 and is_physically_empty:
            tray_type, tray_id = 0, 99  
            
        if db_config:
            if tray_type == 0:
                r_val, c_val = db_config["et_rows"], db_config["et_cols"]
            else:
                r_val, c_val = db_config["ft_rows"], db_config["ft_cols"]
        else:
            r_val, c_val = (5, 8)

        if r_val > c_val:
            rows, cols = c_val, r_val
        else:
            rows, cols = r_val, c_val

        print(f"[INFO] Mode: {'EMPTY' if tray_type == 0 else 'FILLING'} | Part Number: {resolved_part_number} | Matrix Locked: {rows}x{cols}")

        grid, rows, cols = build_grid_kmeans(objects, rows, cols)
        
        occupancy, blades, missing_flag, missing_elements = analyze_grid(grid, tray_type, expected_blade_class)
        tray_fill_status = 4 if tray_type == 5 else (5 if missing_flag else (1 if blades else 0))

        tray_bbox = compute_tray_bbox_from_masks(objects, H, W)
        expected_centers = compute_expected_slot_centers_from_bbox(rows, cols, tray_bbox)
        
        actual_path = render_and_save_overlay(
            img, grid, tray_id, tray_fill_status, expected_centers, SAVE_DIR, valid_ids, corners_meta
        )

        response = build_response(
            occupancy, blades, tray_id, rows, cols, 
            tray_type, tray_fill_status, actual_path, missing_elements, resolved_part_number
        )
        return response