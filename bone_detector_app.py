import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
import gdown  

MODEL_PATH = "bone_fracture_model.h5"
FILE_ID = "16FPtX-0RMlz06KGwOIwV1TEd8mmDbUuo" 
URL = f"https://drive.google.com/file/d/16FPtX-0RMlz06KGwOIwV1TEd8mmDbUuo/view?usp=sharing"

# Download the model if not available locally
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model from Google Drive..."):
        gdown.download(URL, MODEL_PATH, quiet=False)


model = load_model(MODEL_PATH)

# App header
st.header('Bone Fracture Classification (Fractured vs Normal)')

# Define binary labels
class_names = ['Fractured', 'Normal']

# Prediction function
def classify_image(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_expanded = tf.expand_dims(input_image_array, 0)  # Add batch dimension

    predictions = model.predict(input_image_expanded)
    score = predictions[0][0]

    if score >= 0.5:
        label = class_names[1]  # Normal
        confidence = score * 100
    else:
        label = class_names[0]  # Fractured
        confidence = (1 - score) * 100

    outcome = f"The image is classified as **{label}** with a confidence of **{confidence:.2f}%**"
    return outcome

# Upload file
uploaded_file = st.file_uploader('Upload an X-ray Image', type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    # Save the file
    file_path = os.path.join('upload', uploaded_file.name)
    os.makedirs('upload', exist_ok=True)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    # Display the image
    st.image(uploaded_file, width=300, caption="Uploaded X-ray Image")

    # Run prediction
    st.markdown(classify_image(file_path))
