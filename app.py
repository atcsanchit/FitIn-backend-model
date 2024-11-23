import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import os

def list_directories(location):
    try:
        # List all entries in the given location
        entries = os.listdir(location)
        # Filter entries to include only directories
        directories = [entry for entry in entries if os.path.isdir(os.path.join(location, entry))]
        return directories
    except FileNotFoundError:
        print(f"The location '{location}' does not exist.")
        return []
    except PermissionError:
        print(f"Permission denied to access '{location}'.")
        return []

model = tf.keras.models.load_model(r"artifacts\saved_model.h5")

st.title("Exercise Image Classification")

uploaded_image = st.file_uploader("Upload a exercise image", type=["jpg", "png"])

if uploaded_image is not None:

    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Classifying...")
    image = image.resize((224, 224))  
    image_array = np.array(image) / 255.0 
    image_array = np.expand_dims(image_array, axis=0)  

    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction, axis=1)[0]

    result = None
    classes = list_directories("artifacts\data ingestion\FitIn_classes")
    result = classes[predicted_class]
    st.write(f"Disease detected: {result}, classes: {classes}, predicted class: {predicted_class}")
