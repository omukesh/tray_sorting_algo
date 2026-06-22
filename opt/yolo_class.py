from ultralytics import YOLO

# Load the model (this will default to CPU if no GPU is found)
model = YOLO('weights/best.pt') 

# Access the names dictionary
class_names = model.names

# Print the list of classes
print("Total classes:", len(class_names))
for id, name in class_names.items():
    print(f"ID {id}: {name}")
