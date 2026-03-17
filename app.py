import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# ==============================
# 🧠 MODEL LADEN
# ==============================

@st.cache_resource
def load_model():
    # Funktioniert sofort (kein best.pt nötig)
    return YOLO("yolov8n.pt")

model = load_model()

# ==============================
# 🎨 UI
# ==============================

st.title("👕 YOLO Kleidungs-Erkennung")

uploaded_file = st.file_uploader(
    "Bild hochladen",
    type=["jpg", "jpeg", "png"]
)

# ==============================
# 🔍 DETECTION
# ==============================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Originalbild", use_container_width=True)

    img_array = np.array(image)

    # YOLO Prediction
    results = model(img_array)

    result = results[0]

    boxes = result.boxes
    names = model.names

    st.subheader("Erkannte Objekte")

    img_draw = img_array.copy()

    if boxes is not None:

        for box in boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            cls = int(box.cls[0])

            label = names[cls]

            # Bounding Box zeichnen
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)

            text = f"{label} ({confidence:.2f})"

            cv2.putText(
                img_draw,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

            st.write(f"**{label}** – {confidence*100:.1f}%")

    else:
        st.warning("Keine Objekte erkannt.")

    st.image(img_draw, caption="Erkennung", use_container_width=True)

# ==============================
# ℹ️ INFO
# ==============================

st.info("💡 Tipp: Für bessere Kleidungserkennung solltest du ein eigenes YOLO-Modell trainieren (best.pt).")
