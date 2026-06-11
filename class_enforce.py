import cv2
import numpy as np
from ultralytics import YOLO

# ==============================================================================
# 🗃️ 1. CORRECTED ROI BOUNDS (Swapped to match physical layout positions)
# ==============================================================================
# Format: (X, Y, Width, Height)
# Pinpointing the bottom-left marker area for the primary SKU Pointer
ROI_POINTER = (1400, 780, 500, 315)  

SKU_CONFIG = {
    1: {"name": "blade012",   "class_id": 3},
    2: {"name": "blade022",   "class_id": 4},
    3: {"name": "blade052",   "class_id": 6},
    4: {"name": "blade042",   "class_id": 5},
    5: {"name": "blade072",   "class_id": 7},
    8: {"name": "blade22001", "class_id": 8},
    9: {"name": "blade22002", "class_id": 9},
    10: {"name": "blade35032", "class_id": 10},
    11: {"name": "blade35046", "class_id": 11},
}

def detect_pointer_id(img: np.ndarray, roi: tuple) -> int:
    """Crops to ROI, scans using 4x4 dictionary, returns ID or -1 if empty."""
    x, y, w, h = roi
    H, W = img.shape[:2]
    
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)
    
    roi_crop = img[y1:y2, x1:x2]
    if roi_crop.size == 0:
        return -1

    # Initialize 4x4 ArUco library detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    
    corners, ids, _ = detector.detectMarkers(roi_crop)
    
    if ids is not None:
        detected_list = ids.flatten().tolist()
        # Disregard background interference tags like ID 17
        filtered_ids = [uid for uid in detected_list if uid != 17]
        if filtered_ids:
            return filtered_ids[0]
            
    return -1

# ==============================================================================
# 🚀 2. MANDATORY VERIFICATION PIPELINE
# ==============================================================================
def process_strict_roi_mapping(image_path: str):
    model = YOLO("best.pt")
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error: Cannot open image frame at '{image_path}'")
        return

    print(f"\n🔍 Step 1: Mandatorily scanning inside corrected ROI {ROI_POINTER}...")
    active_id = detect_pointer_id(img, ROI_POINTER)
    
    # MANDATORY ENFORCEMENT INTERCEPT: No defaults allowed
    if active_id == -1:
        print("\n🛑 CRITICAL ERROR: No valid 4x4 ArUco marker found inside the scan zone!")
        print("Conveyor halt initiated. Please adjust camera framing or check marker placement.")
        
        # Draw the failed scan zone box on the image so you can see where it looked
        cv2.rectangle(img, (ROI_POINTER[0], ROI_POINTER[1]), 
                      (ROI_POINTER[0]+ROI_POINTER[2], ROI_POINTER[1]+ROI_POINTER[3]), (0, 0, 255), 3)
        cv2.putText(img, "FAILED SCAN ZONE", (ROI_POINTER[0], ROI_POINTER[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imwrite("failed_scan_debug.png", img)
        return

    print(f"✅ Success: 4x4 ArUco marker detected. Verified ID: {active_id}")

    meta = SKU_CONFIG.get(active_id)
    if not meta:
        print(f"❌ Configuration Error: Detected ID {active_id} has no matching rules inside SKU_CONFIG.")
        return

    expected_class_id = meta["class_id"]
    forced_sku_name = meta["name"]
    print(f"🎯 Target Target Locked: Forcing all blade objects to match SKU -> {forced_sku_name}")

    print("\n🧠 Step 2: Running full frame YOLO inference...")
    results = model(image_path, verbose=False)
    
    raw_boxes = []
    raw_confidences = []
    raw_class_ids = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        raw_boxes.append([x1, y1, x2 - x1, y2 - y1])
        raw_confidences.append(float(box.conf[0]))
        raw_class_ids.append(int(box.cls[0]))

    indices = cv2.dnn.NMSBoxes(raw_boxes, raw_confidences, score_threshold=0.25, nms_threshold=0.65)
    
    slots_count = 0
    blades_count = 0

    print("\n🔄 Step 3: Processing overrides and mapping coordinates...")
    if len(indices) > 0:
        for idx in indices.flatten():
            x, y, w, h = raw_boxes[idx]
            cls_id = raw_class_ids[idx]
            conf = raw_confidences[idx]
            
            color = (0, 255, 0)
            display_label = ""

            if cls_id == 2:
                display_label = "slot"
                slots_count += 1
            elif cls_id in range(3, 12):
                display_label = forced_sku_name
                blades_count += 1
            else:
                continue

            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, f"{display_label} {conf:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Highlight the active scan box area on the output image
    cv2.rectangle(img, (ROI_POINTER[0], ROI_POINTER[1]), 
                  (ROI_POINTER[0]+ROI_POINTER[2], ROI_POINTER[1]+ROI_POINTER[3]), (0, 255, 255), 3)
    cv2.putText(img, f"ROI_POINTER SCAN ZONE (ID: {active_id})", (ROI_POINTER[0], ROI_POINTER[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    print("\n" + "="*60)
    print(f"📊 STRICT PIPELINE SUMMARY")
    print(f"📌 ACTIVE ARUCO MAPPED: ID {active_id} ----> SKU TYPE: {forced_sku_name}")
    print("="*60)
    print(f"🔹 TOTAL FOAM BASE SLOTS      : {slots_count}")
    print(f"🔹 TOTAL ENFORCED TARGET BLADES: {blades_count}")
    print("="*60)

    cv2.imwrite("roi_mapped_result.png", img)
    print("💾 Visual feedback saved successfully to 'roi_mapped_result.png'.\n")

if __name__ == "__main__":
    sample_image = "/home/mdl/Projects/tray_algo/input/501.png"
    process_strict_roi_mapping(image_path=sample_image)