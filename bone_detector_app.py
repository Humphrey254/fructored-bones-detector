import os
import keras
from keras.models import load_model
import streamlit as st 
import tensorflow as tf
import numpy as np

# App header
st.header('Bone Fracture Classification (Fractured vs Normal)')

# Load binary classification model
model = load_model('bone_fracture_model.h5')

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
