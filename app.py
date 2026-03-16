import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# ==============================
# MODEL LOAD
# ==============================

@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")   # automatisch geladen
    return model

model = load_model()

# ==============================
# UI
# ==============================

st.title("👕 YOLO KI Kleidungs-Erkennung")

st.write("Lade ein Bild hoch und YOLO erkennt die Kleidungsstücke.")

uploaded_file = st.file_uploader(
    "Bild hochladen",
    type=["jpg","jpeg","png"]
)

# ==============================
# IMAGE PROCESSING
# ==============================

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    img = np.array(image)

    st.image(image, caption="Originalbild", use_container_width=True)

    # YOLO Prediction
    results = model(img)

    result = results[0]

    boxes = result.boxes
    names = model.names

    detected_items = []

    # Bounding Boxes zeichnen
    for box in boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = names[cls]

        detected_items.append(label)

        cv2.rectangle(
            img,
            (x1,y1),
            (x2,y2),
            (0,255,0),
            2
        )

        cv2.putText(
            img,
            label,
            (x1,y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0,255,0),
            2
        )

    st.image(img, caption="Erkannte Kleidung", use_container_width=True)

    if detected_items:

        st.subheader("Erkannte Kleidungsstücke")

        for item in detected_items:
            st.write("•", item)

    else:

        st.warning("Keine Kleidung erkannt.")
