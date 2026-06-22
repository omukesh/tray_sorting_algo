import cv2
import os
import numpy as np

# -------------------------
# SETTINGS
# -------------------------
output_folder = "aruco_4x4_png"
os.makedirs(output_folder, exist_ok=True)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

marker_size = 90
border = 6
text_space = 28   # increased space below marker

total_w = marker_size + (2 * border)
total_h = marker_size + (2 * border) + text_space

# -------------------------
# GENERATE
# -------------------------
for marker_id in range(33):

    marker = np.zeros((marker_size, marker_size), dtype=np.uint8)
    cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size, marker, 1)

    canvas = np.ones((total_h, total_w), dtype=np.uint8) * 255

    # place marker
    canvas[border:border+marker_size, border:border+marker_size] = marker

    # ID text
    text = str(marker_id)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

    text_x = (total_w - text_size[0]) // 2
    text_y = marker_size + border + 20   # more gap here

    cv2.putText(canvas, text,
                (text_x, text_y),
                font,
                font_scale,
                (0,),
                thickness,
                cv2.LINE_AA)

    save_path = os.path.join(output_folder, f"aruco_{marker_id}.png")
    cv2.imwrite(save_path, canvas)

print("All markers saved.")
