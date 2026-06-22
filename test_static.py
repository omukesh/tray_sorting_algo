# test_static.py
import os
import sys
import cv2
from tray_analyzer import TrayAnalyzer

# Force Qt window server logging outputs silent
os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts=false;*.debug=false"
os.environ["LC_ALL"] = "C"

# ── CONFIGURATION PARAMETERS ─────────────────────────────────
MODEL_PATH  = "weights/best.pt"     
TEST_IMAGES = [
    "input/15.jpg",
    # "input/2.png",
    # "input/6.png",
    #"/home/mdl/Documents/model_training/Tray_train_data/tray_final_dataset/test/images/snap_1080p_14-19-18-760.jpg"
]
WINDOW_W, WINDOW_H = 1280, 720
# ────────────────────────────────────────────────────────────

def print_telemetry_report(res: dict) -> None:
    """Renders high-visibility system logging matrices to terminal."""
    mode_string = {0: "EMPTY TRAY", 1: "FILLING TRAY", 5: "ARUCO ERROR"}.get(res["tray_type"], "UNKNOWN")
    
    print("\n" + "═"*70)
    print(f" SYSTEM DETECTION TELEMETRY REPORT")
    print("═"*70)
    print(f"  🔹 Operational Mode : {mode_string}")
    print(f"  🔹 Isolated Tray ID: {res['Tray_ID']}")
    print(f"  🔹 Fill Status Code: {res['tray_fill_status']}")
    print(f"  🔹 Dimension Map   : {res['rows']} Rows x {res['cols']} Columns")
    print(f"  🔹 Mapped Blades   : {res['blade_elements']}")
    if res['missing_elements']:
        print(f" MISSING SLOTS   : {res['missing_elements']}")
    print("─"*70)
    
    # Render column-major grid preview layout maps
    print("  OCCUPANCY MAP PREVIEW (COLUMN-MAJOR):")
    rows, cols = res["rows"], res["cols"]
    grid_data = res["occupancy_grid"]
    
    for c in range(cols):
        slot_indices = []
        occupancy_values = []
        for r_idx in range(rows):
            slot_id = c * rows + r_idx + 1
            slot_indices.append(slot_id)
            occupancy_values.append(grid_data[r_idx][c])
        print(f"   Column {c+1:02d} | Slots {slot_indices} : {occupancy_values}")
    print("═"*70 + "\n")

def run_pipeline_test(image_path: str, analyzer: TrayAnalyzer) -> None:
    """Executes prediction calculations and hooks UI rendering frames."""
    if not os.path.exists(image_path):
        print(f"Error: Targeted test image file does not exist at: '{image_path}'")
        return

    print(f"\n▶ Spinning Frame Pipeline for Target Path: {image_path}")
    try:
        metrics_response = analyzer.analyze_image(image_path)
        print_telemetry_report(metrics_response)
        
        # Display the validation rendering image file back to window loop
        overlay_img = cv2.imread(metrics_response["image_path"])
        if overlay_img is not None:
            window_title = f"Production Line Inspection Frame - Tray ID: {metrics_response['Tray_ID']}"
            cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_title, WINDOW_W, WINDOW_H)
            cv2.imshow(window_title, overlay_img)
            
            print("  [UI] Press any keyboard key on the visual frame window to advance...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    except Exception as e:
        print(f"Pipeline Breakdown Exception: {str(e)}")

if __name__ == "__main__":
    print("[INIT] Loading internal YOLOv8 weight configurations into VRAM core layer...")
    pipeline_analyzer = TrayAnalyzer(MODEL_PATH)
    
    for sample_path in TEST_IMAGES:
        run_pipeline_test(sample_path, pipeline_analyzer)
    print("[COMPLETE] Local validation script evaluation sequences concluded successfully.")