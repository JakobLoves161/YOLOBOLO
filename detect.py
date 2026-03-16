from ultralytics import YOLO
import cv2

# Trainiertes Modell laden
model = YOLO("runs/detect/clothing_detector/weights/best.pt")

image_path = "test.jpg"

results = model(image_path)

for result in results:

    boxes = result.boxes

    for box in boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        label = model.names[class_id]

        print(label, confidence)

    annotated = result.plot()

    cv2.imshow("Detection", annotated)
    cv2.waitKey(0)
