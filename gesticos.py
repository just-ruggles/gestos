import streamlit as st
import cv2
import numpy as np
from PIL import Image as Image, ImageOps as ImagOps
from keras.models import load_model
import platform
import os

st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://wallpapers.com/images/hd/black-carbon-fiber-1biekffyzs37csto.jpg");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Título rojo (usando HTML personalizado)
st.markdown("<h1 style='color: red;'>Reconocimiento de Imágenes a lo Spider-man</h1>", unsafe_allow_html=True)

# Mostrar versión de Python
st.write("Versión de Python:", platform.python_version())

# Verificación de existencia del modelo
model_path = 'keras_model.h5'
if not os.path.exists(model_path):
    st.error("❌ No se encontró el archivo 'keras_model.h5'. Por favor verifica que esté en la misma carpeta que este script.")
    st.stop()

model = load_model(model_path)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Imagen inicial
image = Image.open('peni parker.webp')
st.image(image, width=350)

# Sidebar con subheader azul
with st.sidebar:
    st.markdown("<h3 style='color: blue;'>Usando un modelo entrenado en Teachable Machine</h3>", unsafe_allow_html=True)
    st.write("Puedes usarlo en esta app para identificar gestos.")

# Cámara para tomar foto
img_file_buffer = st.camera_input("Toma una Foto")

if img_file_buffer is not None:
    # Inicializar array para imagen normalizada
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    
    # Cargar imagen de la cámara
    img = Image.open(img_file_buffer)
    img = img.resize((224, 224))
    img_array = np.array(img)

    # Normalizar imagen
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    # Ejecutar predicción
    prediction = model.predict(data)
    print(prediction)

    # Mostrar resultados con headers azules
    if prediction[0][0] > 0.5:
        st.markdown(f"<h2 style='color: blue;'>Izquierda, con Probabilidad: {prediction[0][0]:.2f}</h2>", unsafe_allow_html=True)
    if prediction[0][1] > 0.5:
        st.markdown(f"<h2 style='color: blue;'>Arriba, con Probabilidad: {prediction[0][1]:.2f}</h2>", unsafe_allow_html=True)
    # if prediction[0][2] > 0.5:
    #     st.markdown(f"<h2 style='color: blue;'>Derecha, con Probabilidad: {prediction[0][2]:.2f}</h2>", unsafe_allow_html=True)
