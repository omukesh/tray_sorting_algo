from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO("best.pt")

# Image path
image_path = "/home/mdl/Projects/tray_algo/tray4.png"

# Run inference
results = model(image_path)

# Read image
img = cv2.imread(image_path)

# Blade classes (to be remapped)
valid_classes = list(range(3, 12))

# User input
target_class = int(input("Enter target class (3-11): "))

if target_class not in valid_classes:
    print("Please enter a class between 3 and 11")
    exit()

count = 0

for box in results[0].boxes:

    # Original detected class
    cls_id = int(box.cls[0])

    # Bounding box coordinates
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    # Classes 0,1,2 remain unchanged
    if cls_id in [0, 1, 2]:
        label = model.names[cls_id]

    # Classes 3-11 become the selected blade class
    elif cls_id in valid_classes:
        label = model.names[target_class]

    else:
        continue

    count += 1

    # Draw bounding box
    cv2.rectangle(
        img,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    # Draw label
    cv2.putText(
        img,
        label,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    print(
        f"Original: {model.names[cls_id]}  -->  Displayed As: {label}"
    )

# Save output image
cv2.imwrite("output.png", img)

print(f"\nTotal detections shown: {count}")
print("Output saved as output.png")

# Display output image
cv2.imshow("Output", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
