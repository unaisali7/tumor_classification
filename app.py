import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('cnn_tumor.h5')

# Define the prediction function
def make_prediction(img, model):
    img = img.resize((128, 128))  # Resize image to model input size
    img = np.array(img)  # Convert image to NumPy array
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize pixel values
    res = model.predict(img)  # Get the model's prediction
    return res[0][0]  # Return probability for class

# Set page configuration for better UI
st.set_page_config(page_title="Tumor Classification", page_icon="🧠", layout="centered")

# Add custom CSS to style the UI
st.markdown("""
    <style>
        body {
            background-color: #f0f2f6;
        }
        .stApp {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
        }
        h1 {
            text-align: center;
            color: #1f77b4;
        }
        h2 {
            text-align: center;
            color: #ff7f0e;
        }
        .footer {
            position: fixed;
            bottom: 0;
            width: 100%;
            text-align: center;
            padding: 10px;
            background-color: #f0f2f6;
        }
    </style>
    """, unsafe_allow_html=True)

# Page title and subtitle
st.title("🧠 Tumor Detection CNN Model")
st.write("Upload an MRI scan to check for the presence of a tumor.")

# File uploader for the image
uploaded_file = st.file_uploader("Choose an MRI scan image (JPG format)", type=["jpg", "jpeg", "png"])

# If a file is uploaded
if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded MRI scan", use_column_width=True)

    # Add a submit button
    if st.button("Submit for Classification"):
        st.write("Analyzing the image...")

        # Make prediction
        label = make_prediction(img, model)

        # Display the result with enhanced UI
        if label >= 0.5:
            st.markdown(
                "<h2 style='color: red;'>🚨 Tumor Detected! 🚨</h2>", 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<h2 style='color: green;'>✅ No Tumor Detected!</h2>", 
                unsafe_allow_html=True
            )

# Footer message
st.markdown(
    "<div class='footer'>Built with ❤️ using TensorFlow and Streamlit</div>", 
    unsafe_allow_html=True
)