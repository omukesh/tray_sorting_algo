import cv2
import numpy as np
from ultralytics import YOLO

# ==========================================
# 1. GLOBAL PRODUCTION CONFIGURATION
# ==========================================
# Mapping ArUco Pointer IDs to their expected blade classes
# This acts as your source of truth for Class Enforcement
SKU_CONFIG = {
    1: {"name": "blade012", "class_id": 3, "matrix": (8, 5)},
    2: {"name": "blade022", "class_id": 4, "matrix": (8, 5)},
    3: {"name": "blade052", "class_id": 6, "matrix": (8, 5)},
    4: {"name": "blade042", "class_id": 5, "matrix": (7, 4)},
    5: {"name": "blade072", "class_id": 7, "matrix": (4, 2)},
    8: {"name": "blade198", "class_id": 8, "matrix": (5, 4)},  # Example from repo plan
}

# Universal Functional Classes
CLASS_MAP = {
    0: "slot_empty",     # Exclusive to Empty Tray
    1: "blade_generic",   # Exclusive to Empty Tray
    2: "slot",           # Universal filling tray slot anchor
}

def run_advanced_inference(image_path: str, pointer_id: int, is_empty_tray: bool = False):
    """
    Advanced production pipeline handling automatic class enforcement, 
    Non-Maximum Suppression, and performance logging.
    """
    # Load Model
    model = YOLO("best.pt")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return

    # Run inference
    results = model(image_path)
    
    # Raw components arrays for NMS processing
    boxes = []
    confidences = []
    class_ids = []

    # Parse raw YOLO boxes
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w, h = x2 - x1, y2 - y1
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        
        boxes.append([x1, y1, w, h])
        confidences.append(conf)
        class_ids.append(cls)

    # Apply Non-Maximum Suppression to kill double-bounding boxes
    # score_threshold=0.25, nms_threshold=0.65 to accommodate tightly packed grids
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.25, nms_threshold=0.65)
    
    # Summary Metrics Counters
    counts = {"valid_slots": 0, "valid_blades": 0, "wrong_sku": 0, "anomalies": 0}
    
    # Process only the surviving indices after NMS
    for idx in indices.flatten():
        x, y, w, h = boxes[idx]
        cls_id = class_ids[idx]
        conf = confidences[idx]
        
        # Determine tracking labels based on operational strategy
        color = (0, 255, 0) # Default Green for verified assets
        label = "UNKNOWN"
        is_anomaly = False

        if is_empty_tray:
            # --------------------------------------------------
            # TRACK A: EMPTY TRAY TRACKER (Only allows 0 and 1)
            # --------------------------------------------------
            if cls_id == 0:
                label = "slot_empty"
                counts["valid_slots"] += 1
            elif cls_id == 1:
                label = "⚠️ CRITICAL: blade_generic"
                color = (0, 0, 255) # Red warning for remaining components
                counts["valid_blades"] += 1
            else:
                label = f"WRONG TRACK CLASS: {model.names.get(cls_id, cls_id)}"
                color = (0, 165, 255) # Orange anomaly indicator
                counts["anomalies"] += 1
                
        else:
            # --------------------------------------------------
            # TRACK B: FILLING TRAY SKU ENFORCEMENT (2 and Target Class)
            # --------------------------------------------------
            target_sku_meta = SKU_CONFIG.get(pointer_id)
            if not target_sku_meta:
                print(f"Operational Fail: Pointer ID {pointer_id} unregistered in system config.")
                return

            expected_cls_id = target_sku_meta["class_id"]
            expected_sku_name = target_sku_meta["name"]

            if cls_id == 2:
                label = "slot"
                counts["valid_slots"] += 1
            elif cls_id == expected_cls_id:
                label = expected_sku_name
                counts["valid_blades"] += 1
            elif cls_id in range(3, 12):
                # Detected a different blade SKU entirely!
                label = f"❌ WRONG SKU: {model.names.get(cls_id, 'Unknown')}"
                color = (255, 0, 255) # Magenta highlight for immediate operator intervention
                counts["wrong_sku"] += 1
            else:
                label = f"ANOMALY: {model.names.get(cls_id, cls_id)}"
                color = (0, 165, 255)
                counts["anomalies"] += 1

        # Render Overlays
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, f"{label} {conf:.2f}", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Print Production Telemetry Matrix
    print("\n" + "="*50)
    print(f"📊 SYSTEM ENGINE REPORT | MODE: {'EMPTY TRAY' if is_empty_tray else 'FILLING TRAY'}")
    print(f"📌 ACTIVE POINTER ID REFERENCE: {pointer_id}")
    print("="*50)
    for metric, val in counts.items():
        print(f"🔹 {metric.upper().replace('_', ' ')}: {val}")
    print("="*50)

    # Save Output
    cv2.imwrite("tray_output/enforced_output.png", img)
    print("💾 Analysis rendered successfully onto 'enforced_output.png'.")

# ==========================================
# 3. PRODUCTION INFERENCE TEST SIMULATOR
# ==========================================
if __name__ == "__main__":
    image_path = "/home/mdl/Projects/tray_algo/input/tray4.png"
    
    # Simulate a system check on a Filling Tray assigned to ArUco Pointer ID #3 
    # This automatically locks expectations down to look for 'blade052' (Class #6) and 'slot' (Class #2)
    run_advanced_inference(image_path=image_path, pointer_id=3, is_empty_tray=False)