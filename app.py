import streamlit as st 
from PIL import Image
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np 
from keras.models import load_model 
import os

# Load the trained model
model = load_model("tuberculosis.h5")

# Label mapping
labels = {0: 'No Tuberculosis', 1: 'Tuberculosis'}
tuberculosis_set = {'Tuberculosis'}

# Prediction function
def processed_img(img_path):
    img = load_img(img_path, target_size=(224, 224, 3))
    img = img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    y_class = prediction.argmax(axis=-1)
    result = labels[int(y_class)]
    return result.capitalize()

# Main Streamlit app
def run():
    # Page title and description
    st.markdown("<h1 style='text-align: center; color: #0a9396;'>🫁 TB-EnsembleX: Tuberculosis Detection </h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 18px;'>An ensemble transfer learning model for high-accuracy TB screening from chest X-rays</p>", unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar with structured layout
    with st.sidebar:
        st.header("🧠 Project Info")
        st.markdown("""
        This application uses deep learning models to detect **tuberculosis** from **chest X-ray** images.
        Powered by **Convolutional Neural Networks (CNNs)** and **Transfer Learning** trained on real medical data.
        """)

        with st.expander("🔍 Model Details"):
            st.markdown("""
            - Ensemble of **VGG16, VGG19, InceptionV3, Xception**
            - Features concatenated and reduced via **PCA**
            - **SMOTE** applied to balance class distribution
            - Final classifier: **Voting-based Logistic Regression Ensemble**
            - Achieved **99% accuracy**
            """)

        with st.expander("📂 Classes Detected"):
            st.markdown("""
            - ✅ **No Tuberculosis**
            - 🚨 **Tuberculosis**
            """)

        with st.expander("📁 Dataset Info"):
            st.markdown("""
            - Real-world **Chest X-ray datasets**
            - Preprocessed and resized to 224x224
            - Balanced using **SMOTE**
            - Augmented with standard transformations
            """)

        st.markdown("---")
        st.markdown("👨‍💻 **Developed by:** Akshwin T")
        st.markdown("📬 **Contact:** [akshwint.2003@gmail.com](mailto:akshwint.2003@gmail.com)")

    # File uploader
    st.subheader("📤 Upload a Chest X-Ray Image:")
    img_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

    # Process image
    if img_file is not None:
        img = Image.open(img_file).resize((250, 250))
        st.image(img, caption="Uploaded Image", use_container_width=300)

        # Save image locally
        upload_dir = "./upload_image"
        os.makedirs(upload_dir, exist_ok=True)
        save_path = os.path.join(upload_dir, img_file.name)
        with open(save_path, "wb") as f:
            f.write(img_file.getbuffer())

        # Predict and display result
        result = processed_img(save_path)
        st.markdown("---")
        if result in tuberculosis_set:
            st.error("🚨 **TUBERCULOSIS DETECTED!** Please consult a medical professional.")
        else:
            st.success("✅ **NO TUBERCULOSIS DETECTED!**")

# Run the app
if __name__ == "__main__":
    run()