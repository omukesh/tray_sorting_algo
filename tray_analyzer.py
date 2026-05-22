# tray_analyzer.py
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from tray_helpers import (
    SAVE_DIR, analyze_grid, build_grid_kmeans, build_response, detect_aruco_ids_and_first_corners,
    decide_layout_and_primary_id, extract_objects_from_result, TRAY_LAYOUTS_FILLING, TRAY_LAYOUTS_EMPTY,
    infer_empty_layout_dynamic, compute_tray_bbox_from_masks, compute_expected_slot_centers_from_bbox, 
    render_and_save_overlay
)

class TrayAnalyzer:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def analyze_image(self, image_path: str):
        img = cv2.imread(image_path)
        if img is None: 
            raise FileNotFoundError(f"Cannot read image file: {image_path}")
        H, W = img.shape[:2]

        # 1. ARUCO LAYER
        valid_ids, corners = detect_aruco_ids_and_first_corners(img)
        tray_id, tray_type = decide_layout_and_primary_id(valid_ids)
        
        # 2. YOLO LAYER WITH HIGH PRECISION MASK RESIZING
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
                # Use Nearest-Neighbor interpolation to keep binary lines pixel-accurate
                mask_resized = cv2.resize(mask_raw, (W, H), interpolation=cv2.INTER_NEAREST)
                det["mask_array"] = (mask_resized > 0.5).astype("uint8")
            else:
                det["mask_array"] = None
            detections.append(det)

        # 3. OBJECT EXTRACTION
        objects = extract_objects_from_result(detections, H, W)
        is_physically_empty = all(o["cls"] in ["slot_empty", "blade_generic"] for o in objects)
        
        # 4. HYBRID ENGINE & EXCLUSIVE MAPPING INTERACTION
        expected_blade_class = ""
        
        if tray_type == 5 and is_physically_empty:
            tray_type, tray_id = 0, 99  # Forced fallback
            
        if tray_type == 0:
            expected_blade_class = "blade_generic"
            if tray_id in TRAY_LAYOUTS_EMPTY:
                rows, cols = TRAY_LAYOUTS_EMPTY[tray_id]
            else:
                rows, cols = infer_empty_layout_dynamic(len(objects))
        else:
            # Filling tray mapping lookup
            layout_data = TRAY_LAYOUTS_FILLING.get(tray_id, (5, 8, "unknown_blade"))
            rows, cols = layout_data[0], layout_data[1]
            expected_blade_class = layout_data[2]

        print(f"[INFO] Mode: {'EMPTY' if tray_type == 0 else 'FILLING'} | Tray_ID: {tray_id} | Inferred Dimension: {rows}x{cols}")

        # 5. KMEANS GRID MAPPING
        grid, rows, cols = build_grid_kmeans(objects, rows, cols)
        
        # 6. EXCLUSIVE ALGO ANALYSIS ENGINE
        occupancy, blades, widths, missing_flag, missing_elements = analyze_grid(grid, tray_type, expected_blade_class)
        
        # Status configurations
        tray_fill_status = 4 if tray_type == 5 else (5 if missing_flag else (1 if blades else 0))

        # 7. GENERATE EXPECTED GEOMETRIC CENTERS
        tray_bbox = compute_tray_bbox_from_masks(objects, H, W)
        expected_centers = compute_expected_slot_centers_from_bbox(rows, cols, tray_bbox)
        
        # 8. RENDER OVERLAY
        actual_path = render_and_save_overlay(
            img, grid, tray_id, tray_fill_status, expected_centers, SAVE_DIR, valid_ids, corners
        )

        return build_response(
            occupancy, blades, widths, tray_id, rows, cols, 
            tray_type, tray_fill_status, actual_path, missing_elements
        )