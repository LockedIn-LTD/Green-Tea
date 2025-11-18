from ultralytics import YOLO

# Load a YOLO11n PyTorch model
model = YOLO("best.pt")

# Export the model to TensorRT
model.export(format="engine")  # creates 'yolo11n.engine'