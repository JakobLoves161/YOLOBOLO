import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# ==============================
# MODEL LOAD
# ==============================

@st.cache_resource
def load_model():
    # eigenes Modell: "best.pt"
    # oder fallback: yolov8n.pt
    return YOLO("best.pt")

model = load_model()

# ==============================
# UI
# ==============================

st.title("👕 YOLO Kleidungs-Erkennung")

uploaded_file = st.file_uploader("Bild hochladen", type=["jpg","jpeg","png"])

# ==============================
# DETECTION
# ==============================

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Originalbild", use_container_width=True)

    img_array = np.array(image)

    # YOLO Prediction
    results = model(img_array)

    result = results[0]

    boxes = result.boxes
    names = model.names

    st.subheader("🔍 Erkannte Kleidung")

    # OpenCV für Zeichnen
    img_draw = img_array.copy()

    for box in boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])

        label = names[cls]

        # Bounding Box zeichnen
        cv2.rectangle(img_draw, (x1,y1), (x2,y2), (0,255,0), 2)

        text = f"{label} ({conf:.2f})"

        cv2.putText(
            img_draw,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,255,0),
            2
        )

        st.write(f"**{label}** – {conf*100:.1f}%")

    st.image(img_draw, caption="Erkannte Objekte", use_container_width=True)
