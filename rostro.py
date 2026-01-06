import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Detección de Rostros", layout="centered")
st.title("🙂 Detección de Rostros con OpenCV")

# ---------------- CARGA SEGURA DEL CLASIFICADOR ----------------
cascade_path = "haarcascade_frontalface_default.xml"

if not os.path.exists(cascade_path):
    st.error("❌ No se encontró el archivo haarcascade_frontalface_default.xml")
    st.stop()

face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    st.error("❌ El clasificador Haar se cargó vacío.")
    st.markdown("""
    **Posibles causas:**
    - OpenCV mal instalado
    - Archivo XML corrupto
    - Incompatibilidad de versiones
    """)
    st.stop()

# ---------------- FUNCIÓN DE DETECCIÓN ----------------
def detect_faces(image: Image.Image):
    img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return Image.fromarray(img), len(faces)

# ---------------- INTERFAZ STREAMLIT ----------------
uploaded_file = st.file_uploader(
    "Sube una imagen",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)

    st.image(image, caption="Imagen original", use_column_width=True)
    st.info("Procesando imagen...")

    result_img, count = detect_faces(image)

    st.image(
        result_img,
        caption=f"Rostros detectados: {count}",
        use_column_width=True
    )
