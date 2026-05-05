# test_static.py  — run with:  python3 test_static.py
import os, sys

# Suppress Qt font / logging noise BEFORE cv2 loads
os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts=false;*.debug=false"
os.environ["LC_ALL"] = "C"

import cv2
import json
from tray_analyzer import TrayAnalyzer

# ── config ──────────────────────────────────────────────────
MODEL_PATH  = "weights/best.pt"     
TEST_IMAGES = [
    "input/tray4.png",
     #"input/6.png",
    # "input/empty_tray.png",
]
WINDOW_W, WINDOW_H = 900, 720
# ────────────────────────────────────────────────────────────

analyzer = TrayAnalyzer(MODEL_PATH)


def print_matrix(r: dict) -> None:
    mode = {0: "EMPTY TRAY", 1: "FILLING TRAY", 5: "ARUCO ERROR"}.get(
        r["tray_type"], "UNKNOWN")
    print(f"\n{'─'*60}")
    print(f"  Tray_ID={r['Tray_ID']}  Mode={mode}  "
          f"fill_status={r['tray_fill_status']}  "
          f"count={r['count']}")
    print(f"  blades={r['blade_elements']}")
    print(f"  missing_elements={r['missing_elements']}")
    print(f"  widths={r['top_view_widths']}")
    print(f"{'─'*60}")
    print(f"  rows={r['rows']}  cols={r['cols']}")
    print()
    occ  = r["occupancy_grid"]
    rows = r["rows"]
    cols = r["cols"]
    print("\n  --- OCCUPANCY (COLUMN-MAJOR) ---\n")

    for c in range(cols):
        ids = []
        vals = []

        for rb in range(rows):
            sid = c * rows + rb + 1
            ids.append(sid)
            vals.append(occ[rb][c])

        print(f"  {ids}  :  {vals}")
    print()
    if r.get("tray_type") == 0 and r.get("blade_elements"):
        print(f"  ⚠️  BLADE(S) IN EMPTY TRAY — slots: {r['blade_elements']}")
        print(f"     → Operator must remove blade(s) before proceeding.\n")


def run_test(image_path: str) -> None:
    print(f"\n▶  {image_path}")
    result = analyzer.analyze_image(image_path)
    print_matrix(result)
    print(f"  Overlay saved → {result['image_path']}")

    overlay = cv2.imread(result["image_path"])
    if overlay is not None:
        win = "Tray Inspection"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, WINDOW_W, WINDOW_H)
        cv2.imshow(win, overlay)
        print("  Press any key to continue …")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    for path in TEST_IMAGES:
        run_test(path)