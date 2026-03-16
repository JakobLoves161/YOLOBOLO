import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw

# Modell laden
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.title("👕 YOLO Kleidungs-Erkennung")

uploaded_file = st.file_uploader(
    "Bild hochladen",
    type=["jpg","jpeg","png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Originalbild")

    results = model(image)

    result = results[0]

    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()

    draw = ImageDraw.Draw(image)

    detected = []

    for box, cls in zip(boxes, classes):

        x1, y1, x2, y2 = box

        label = model.names[int(cls)]

        detected.append(label)

        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        draw.text((x1, y1), label, fill="red")

    st.image(image, caption="Erkannte Objekte")

    if detected:
        st.subheader("Erkannte Kleidung")
        for d in detected:
            st.write("•", d)
    else:
        st.warning("Keine Kleidung erkannt")
