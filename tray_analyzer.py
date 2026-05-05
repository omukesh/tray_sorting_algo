from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from tray_helpers import (
    SAVE_DIR, analyze_grid, build_grid_kmeans, build_response, detect_aruco_ids_and_first_corners,
    decide_layout_and_primary_id, extract_objects_from_result, TRAY_LAYOUTS,
    compute_tray_bbox_from_masks, compute_expected_slot_centers_from_bbox, render_and_save_overlay
)

class TrayAnalyzer:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def analyze_image(self, image_path: str):
        img = cv2.imread(image_path)
        if img is None: raise FileNotFoundError(image_path)
        H, W = img.shape[:2]

        valid_ids, corners = detect_aruco_ids_and_first_corners(img)
        tray_id, tray_type = decide_layout_and_primary_id(valid_ids)
        
        results = self.model(img, verbose=False)[0]
        detections = []
        
        for i, box in enumerate(results.boxes):
            det = {
                "box": box.xyxy[0].tolist(),
                "name": results.names[int(box.cls)],
                "confidence": float(box.conf[0])
            }
            
            if results.masks is not None:
                
                mask_raw = results.masks.data[i].cpu().numpy()
                mask_resized = cv2.resize(mask_raw, (W, H), interpolation=cv2.INTER_NEAREST)
                mask_binary = (mask_resized > 0.5).astype("uint8")
                
                kernel = np.ones((2, 2), np.uint8)
                mask_tight = cv2.erode(mask_binary, kernel, iterations=1)
                
                det["mask_array"] = mask_tight
            detections.append(det)

        objects = extract_objects_from_result(detections, H, W)
        is_physically_empty = all(o["cls"] in ["slot_empty", "blade_generic"] for o in objects)
        
        if tray_type == 5 and is_physically_empty:
            tray_type, tray_id = 0, 99
            
        if tray_type == 0:
            rows, cols = (6, 9) if len(objects) >= 46 else (5, 8)
        else:
            rows, cols = TRAY_LAYOUTS.get(tray_id, (5, 8))

        grid, rows, cols = build_grid_kmeans(objects, rows, cols)
        occupancy, blades, widths, missing_flag, missing_elements = analyze_grid(grid)
        
        tray_fill_status = 4 if tray_type == 5 else (5 if missing_flag else (1 if blades else 0))

        tray_bbox = compute_tray_bbox_from_masks(objects, H, W)
        expected_centers = compute_expected_slot_centers_from_bbox(rows, cols, tray_bbox)
        
        actual_path = render_and_save_overlay(img, grid, tray_id, tray_fill_status, expected_centers, SAVE_DIR, valid_ids, corners)

        return build_response(occupancy, blades, widths, tray_id, rows, cols, tray_type, tray_fill_status, actual_path, missing_elements)