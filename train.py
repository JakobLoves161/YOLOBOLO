from ultralytics import YOLO

# YOLO Basis Modell laden
model = YOLO("yolov8n.pt")

# Training starten
model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="clothing_detector"
)
