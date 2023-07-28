import os
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.utils import disable_interactive_logging

disable_interactive_logging()

# Function to load a pre-trained model
def load_pretrained_model():
    # Replace 'path_to_your_model' with the actual path to your pre-trained model
    model = tf.keras.models.load_model('model.h5')
    return model

# Function to load a user-uploaded model
def load_user_model(uploaded_model):
    model = tf.keras.models.load_model(uploaded_model.name)
    return model

# Function to predict the image class and probabilities
def predict_image_class_and_probs(model, image_path, classes):
    img = load_img(image_path, target_size=(32, 32, 3))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction, axis=1)[0]
    class_label = classes[class_index]
    probabilities = prediction[0]

    return class_label, probabilities

def main():
    st.title("Skin Cancer Image Classifier")
    st.write("This app predicts the type of skin lesion in the uploaded image.")

    # Upload image file
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Upload model file
    uploaded_model = st.file_uploader("Upload your model (.h5 file)...", type=["h5"])

    if uploaded_file is not None and uploaded_model is not None:
        # Load the model (user-uploaded or pre-trained)
        if uploaded_model.type == "h5":
            model = load_user_model(uploaded_model)
            st.write("Custom model loaded.")
        else:
            model = load_pretrained_model()
            st.write("Pre-trained model loaded.")

        # Mapping of class indices to class labels
        classes = {
            4: 'melanocytic nevi',
            6: 'melanoma',
            2: 'benign keratosis-like lesions',
            1: 'basal cell carcinoma',
            5: 'vascular lesions',
            0: "Bowen's disease",
            3: 'dermatofibroma',
        }

        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Add a button to trigger prediction
        if st.button("Predict"):
            # Predict the image class and probabilities
            class_prediction, probabilities = predict_image_class_and_probs(model, uploaded_file, classes)

            st.subheader("Prediction Result")
            st.write("Predicted Lesion Type: ", class_prediction)
            st.write("Probabilities:")
            for i, prob in enumerate(probabilities):
                st.write(f"{classes[i]}: {prob:.4f}")

            st.write("Uploaded Image Name: ", uploaded_file.name)

if __name__ == "__main__":
    main()