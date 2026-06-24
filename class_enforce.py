import cv2
import numpy as np
from ultralytics import YOLO

# ==============================================================================
#  1. PRODUCTION SKU CONFIGURATION MATRIX (Aligned & Bug-Free)
# ==============================================================================
# Precision cropping zone for the bottom-left primary SKU Pointer marker
ROI_POINTER = (1400, 780, 500, 315)  

# Keys match the physical ArUco ID tag numbers.
# class_id matches the exact model index inside MODEL_NAMES.
SKU_CONFIG = {
    1:  {"name": "blade_hpcr012",    "class_id": 3},
    2:  {"name": "blade_hpcr022",    "class_id": 4},
    4:  {"name": "blade_hpcr042",    "class_id": 5},
    3:  {"name": "blade_hpcr052",    "class_id": 6},
    7:  {"name": "blade_hpcr072",    "class_id": 7},
    10: {"name": "blade_hpcs001",    "class_id": 8},
    11: {"name": "blade_hpcs002",    "class_id": 9},
    13: {"name": "blade_hpcs004",    "class_id": 10},
    14: {"name": "blade_hpcs005",    "class_id": 11},
    15: {"name": "blade_hpcs006",    "class_id": 12},
    16: {"name": "blade_hpcs007",    "class_id": 13},
    18: {"name": "blade_hpcs008",    "class_id": 14},
    19: {"name": "blade_hpcs009",    "class_id": 15},
    20: {"name": "blade_hpcs011",    "class_id": 16},
    30: {"name": "blade_hptr020",    "class_id": 17},
    22: {"name": "blade_lpcr046",    "class_id": 18}, 
    23: {"name": "blade_lpcr35032",  "class_id": 19},
    24: {"name": "blade_lpcr35046",  "class_id": 20}, 
    31: {"name": "blade_lptr050",    "class_id": 21}
}

def detect_pointer_id(img: np.ndarray, roi: tuple) -> int:
    """Crops to pointer zone, runs DICT_4X4_50 scan, returns first non-noise ID."""
    x, y, w, h = roi
    H, W = img.shape[:2]
    
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)
    
    roi_crop = img[y1:y2, x1:x2]
    if roi_crop.size == 0:
        return -1

    # Instantiate specialized 4x4 ArUco reader setup
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    
    corners, ids, _ = detector.detectMarkers(roi_crop)
    
    if ids is not None:
        detected_list = ids.flatten().tolist()
        # Drop background noise tags like ID 17 inside the crop logic phase
        filtered_ids = [uid for uid in detected_list if uid != 17]
        if filtered_ids:
            return filtered_ids[0]
            
    return -1

# ==============================================================================
#  2. CLEAN CORE ENFORCEMENT ENGINE
# ==============================================================================
def process_new_classes_mapping(image_path: str):
    model = YOLO("best.pt")
    img = cv2.imread(image_path)
    if img is None:
        print(f" Error: Cannot open target image frame at '{image_path}'")
        return

    print(f"\n Step 1: Performing mandatory scan within ROI_POINTER box {ROI_POINTER}...")
    active_id = detect_pointer_id(img, ROI_POINTER)
    
    # STRICT BLOCK INTERCEPT: Execution halts completely if ArUco scan fails
    if active_id == -1:
        print("\n CRITICAL COMPLIANCE ERROR: No valid 4x4 ArUco tag discovered inside scan bounds!")
        print("Halting process tracking to prevent unmapped engine runs.")
        
        # Output an error visual block to quickly verify camera framing alignment
        cv2.rectangle(img, (ROI_POINTER[0], ROI_POINTER[1]), 
                      (ROI_POINTER[0]+ROI_POINTER[2], ROI_POINTER[1]+ROI_POINTER[3]), (0, 0, 255), 3)
        cv2.putText(img, "FAILED SCAN WINDOW", (ROI_POINTER[0], ROI_POINTER[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imwrite("failed_scan_debug.png", img)
        return

    print(f" Success: 4x4 ArUco marker isolated. Verified ID: {active_id}")

    meta = SKU_CONFIG.get(active_id)
    if not meta:
        print(f" Configuration Error: Detected ID {active_id} matches no schema options inside SKU_CONFIG.")
        return

    forced_sku_name = meta["name"]
    print(f" Strategy Locked: Mapping all blade detections to class SKU -> {forced_sku_name}")

    print("\n Step 2: Executing deep frame parsing...")
    results = model(image_path, verbose=False)
    
    raw_boxes = []
    raw_confidences = []
    raw_class_ids = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        raw_boxes.append([x1, y1, x2 - x1, y2 - y1])
        raw_confidences.append(float(box.conf[0]))
        raw_class_ids.append(int(box.cls[0]))

    # Apply NMS calculation loop to wipe out layered bounding duplicates
    indices = cv2.dnn.NMSBoxes(raw_boxes, raw_confidences, score_threshold=0.25, nms_threshold=0.65)
    
    slots_count = 0
    blades_count = 0

    print("\n" + "═"*60)
    print("Step 3: Running Global Mapping Translation...")
    print("═"*60)
    
    if len(indices) > 0:
        for idx in indices.flatten():
            x, y, w, h = raw_boxes[idx]
            cls_id = raw_class_ids[idx]
            conf = raw_confidences[idx]
            
            color = (0, 255, 0) # Production Green for system compliance
            display_label = ""

            # Class ID #2 remains the infrastructure base slot foam background
            if cls_id == 2:
                display_label = "slot"
                slots_count += 1
            
            # ALL BLADE VARIANTS (Classes 3 to 21) are cleanly remapped without logs
            elif cls_id in range(3, 22):
                display_label = forced_sku_name
                blades_count += 1
            else:
                continue

            # Overlay bounding indicators back to the raw image frame
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, f"{display_label} {conf:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Highlight active scanner zone boundaries onto output image array
    cv2.rectangle(img, (ROI_POINTER[0], ROI_POINTER[1]), 
                  (ROI_POINTER[0]+ROI_POINTER[2], ROI_POINTER[1]+ROI_POINTER[3]), (0, 255, 255), 3)
    cv2.putText(img, f"ROI_POINTER VALIDATION AREA (ID: {active_id})", (ROI_POINTER[0], ROI_POINTER[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    print("\n" + "="*60)
    print(f"SYSTEM ENGINE PERFORMANCE TELEMETRY")
    print(f" ARUCO LOOKUP KEY DETECTED : ID {active_id} ──> RULE SET: {forced_sku_name}")
    print("="*60)
    print(f"🔹 TOTAL FOAM BASE SLOTS      : {slots_count}")
    print(f"🔹 TOTAL TARGET BLADES MAPPED : {blades_count}")
    print("="*60)

    cv2.imwrite("roi_mapped_result.png", img)
    print(" Process complete. Clean validation frame rendered safely to 'roi_mapped_result.png'.\n")

if __name__ == "__main__":
    # Test path setting configuration
    sample_image = "/home/mdl/Videos/mukesh/tray_sorting_algo/input/7.png"
    process_new_classes_mapping(image_path=sample_image)
