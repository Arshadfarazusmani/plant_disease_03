import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model outside the prediction function to avoid reloading on every prediction
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/PlantDiseaseModel.keras")

model = load_model()

# Model Prediction Function
def model_prediction(test_image):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# Aesthetics and Styling
st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿", layout="wide")

# Custom CSS
st.markdown(
    """
    <style>
    .big-font { font-size:24px !important; }
    .sidebar .sidebar-content { background-color: #f0f8ff; }
    .stButton > button { background-color: #4CAF50; color: white; }
    .stSuccess { background-color: #e6f7e6; padding: 20px; border-radius: 5px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Select Page", ["Home", "Disease Recognition", "About"])

# Home Page
if app_mode == "Home":
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("images/home_page.jpeg", use_column_width=True)
    with col2:
        st.title("Welcome to Plant Disease Detector 🌿")
        st.markdown(
            """
            Our mission is to empower farmers and gardeners with quick and accurate plant disease detection. 
            Upload an image, and our AI will analyze it to identify potential diseases, helping you protect your crops and maintain a healthy garden.
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            ### How It Works:
            1. **Upload Image:** Navigate to the 'Disease Recognition' page and upload a plant image.
            2. **Analysis:** Our system processes the image using deep learning.
            3. **Results:** Receive instant results with disease identification.
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            ### Why Choose Us:
            - **Accuracy:** Advanced AI for reliable detection.
            - **Ease of Use:** User-friendly interface.
            - **Efficiency:** Fast results for timely action.
            """,
            unsafe_allow_html=True,
        )

# About Page
elif app_mode == "About":
    st.title("About the Project")
    st.markdown(
        """
        #### Dataset Information:
        This application uses a dataset of approximately 87,000 RGB images of healthy and diseased plant leaves, 
        categorized into 38 different classes. The dataset was created with offline augmentation from an original dataset, 
        available on GitHub. It's divided into training (80%) and validation (20%) sets, with a separate test set of 33 images.
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        #### Dataset Structure:
        - **Training Set:** 70,295 images
        - **Test Set:** 33 images
        - **Validation Set:** 17,572 images
        """,
        unsafe_allow_html=True,
    )

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.title("Detect Plant Diseases")
    test_image = st.file_uploader("Upload an Image:", type=["png", "jpg", "jpeg"])
    if test_image is not None:
        image = Image.open(test_image)
        st.image(image, caption="Uploaded Image.", use_column_width=True)
        if st.button("Predict"):
            result_index = model_prediction(test_image)
            class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                          'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                          'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                          'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                          'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                          'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                          'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                          'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                          'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                          'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                          'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                          'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                          'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                          'Tomato___healthy']
            st.success(f"Prediction: {class_name[result_index]}")