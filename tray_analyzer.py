# tray_analyzer.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
from ultralytics import YOLO

from tray_helpers import (
    SAVE_DIR,
    analyze_grid,
    build_grid_kmeans,
    build_response,
    compute_expected_slot_centers_from_bbox,
    compute_tray_bbox_from_masks,
    decide_layout_and_primary_id,
    detect_aruco_ids_and_first_corners,
    extract_objects_from_result,
)

_BASE = Path(__file__).resolve().parent
MODEL_PATH_DEFAULT: str = str(_BASE / "models" / "tray_cam.pt")


class TrayAnalyzer:

    def __init__(self, model_path: str = MODEL_PATH_DEFAULT) -> None:
        self.model: YOLO = YOLO(model_path)

    # ─────────────────────────────────────────────

    def analyze_image(self, image_path: str) -> Dict:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot open image: {image_path}")
        return self._analyze(img)

    # ─────────────────────────────────────────────

    def _analyze(self, img: np.ndarray) -> Dict:
        H, W = img.shape[:2]

        # --------------------------------------------------
        # 1. ARUCO → layout
        # --------------------------------------------------
        ids, corners = detect_aruco_ids_and_first_corners(img)

        rows, cols, tray_id, tray_type = decide_layout_and_primary_id(ids)

        tray_fill_status: Optional[int] = 4 if not ids else None

        # --------------------------------------------------
        # 2. YOLO
        # --------------------------------------------------
        res = self.model(img, verbose=False)[0]

        detections = []
        has_masks = res.masks is not None

        for i, box in enumerate(res.boxes):
            det = {
                "box": box.xyxy[0].tolist(),
                "name": res.names[int(box.cls)],
                "confidence": float(box.conf[0])
            }

            if has_masks:
                mask = res.masks.data[i].cpu().numpy()
                mask = cv2.resize(mask, (W, H))
                det["mask_array"] = (mask > 0.5).astype("uint8")
            else:
                det["mask_array"] = None

            detections.append(det)

        # --------------------------------------------------
        # 3. CLASS COUNT DEBUG
        # --------------------------------------------------
        class_counts = {}

        for d in detections:
            cls = d["name"]
            class_counts[cls] = class_counts.get(cls, 0) + 1

        print("\n--- YOLO CLASS COUNTS ---")
        for k, v in class_counts.items():
            print(f"{k}: {v}")
        print(f"TOTAL: {sum(class_counts.values())}")

        # --------------------------------------------------
        # 4. OBJECT EXTRACTION
        # --------------------------------------------------
        objs = extract_objects_from_result(detections, H, W)

        # --------------------------------------------------
        # 5. GRID ASSIGNMENT
        # --------------------------------------------------
        grid, rows, cols = build_grid_kmeans(objs, rows, cols)

        # --------------------------------------------------
        # 6. ANALYSIS
        # --------------------------------------------------
        occ, blades, widths, missing = analyze_grid(grid)

        # --------------------------------------------------
        # 7. STATUS
        # --------------------------------------------------
        if tray_fill_status != 4:
            tray_fill_status = 5 if missing else (1 if blades else 0)

        # --------------------------------------------------
        # 8. EXPECTED CENTERS
        # --------------------------------------------------
        tray_bbox = compute_tray_bbox_from_masks(objs, H, W)
        expected_centers = compute_expected_slot_centers_from_bbox(
            rows, cols, tray_bbox
        )

        # --------------------------------------------------
        # 9. RENDER
        # --------------------------------------------------
        from tray_helpers import render_and_save_overlay

        image_path = render_and_save_overlay(
            img,
            grid,
            tray_id,
            tray_fill_status,
            expected_centers,
            SAVE_DIR,
        )

        # --------------------------------------------------
        # 10. RESPONSE
        # --------------------------------------------------
        return build_response(
            occ,
            blades,
            widths,
            tray_id,
            rows,
            cols,
            tray_type,
            tray_fill_status,
            image_path,
        )