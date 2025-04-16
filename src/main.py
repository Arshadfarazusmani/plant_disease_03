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

# Custom CSS (Modern Streamlit approach)
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
with st.sidebar:
    st.title("Navigation")
    app_mode = st.radio("Select Page", ["Home", "Disease Recognition", "About"])

# Home Page
if app_mode == "Home":
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("images/home_page.jpeg", use_container_width=True)
    with col2:
        st.title("Welcome to Plant Disease Detector 🌿")
        st.markdown(
            """
            Our mission is to empower farmers and gardeners with quick and accurate plant disease detection.
            Upload an image, and our AI will analyze it to identify potential diseases, helping you protect your crops and maintain a healthy garden.
            """
        )
        st.markdown(
            """
            ### How It Works:
            1. **Upload Image:** Navigate to the 'Disease Recognition' page and upload a plant image.
            2. **Analysis:** Our system processes the image using deep learning.
            3. **Results:** Receive instant results with disease identification.
            """
        )
        st.markdown(
            """
            ### Why Choose Us:
            - **Accuracy:** Advanced AI for reliable detection.
            - **Ease of Use:** User-friendly interface.
            - **Efficiency:** Fast results for timely action.
            """
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
        """
    )
    st.markdown(
        """
        #### Dataset Structure:
        - **Training Set:** 70,295 images
        - **Test Set:** 33 images
        - **Validation Set:** 17,572 images
        """
    )

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.title("Detect Plant Diseases")
    test_image = st.file_uploader("Upload an Image:", type=["png", "jpg", "jpeg"])
    if test_image is not None:
        image = Image.open(test_image)
        st.image(image, caption="Uploaded Image.", use_container_width=True)
        if st.button("Predict"):
            result_index = model_prediction(test_image)
            class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                          'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                          'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                          'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                          'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                          'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                          'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                          'Potato Early blight', 'Potato___Late_blight', 'Potato___healthy',
                          'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                          'Not Recognised', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                          'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                          'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                          'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                          'Tomato___healthy']
            diagnosis_info = {

                'Apple___Apple_scab': "Apple scab is a fungal disease that causes dark, olive-green to black spots on leaves and fruit. Treat with fungicides labeled for apple scab. Ensure good air circulation and remove fallen leaves to reduce overwintering spores.",
                'Apple___Black_rot': "Black rot causes dark lesions on leaves and fruit. Remove infected parts promptly and apply fungicides preventatively, especially during wet periods. Prune for better air circulation and sanitation.",
                'Apple___Cedar_apple_rust': "Cedar apple rust results in yellow-orange spots on apple leaves and fruit. Remove nearby cedar trees (the alternate host) if possible. Use fungicides on apple trees before and after bloom.",
                'Apple___healthy': "The apple appears healthy. Monitor for signs of disease and maintain good cultural practices.",
                'Blueberry___healthy': "The blueberry appears healthy. Ensure proper soil pH, irrigation, and fertilization.",
                'Cherry_(including_sour)___Powdery_mildew': "Powdery mildew leads to white, powdery growth on leaves. Treat with fungicides labeled for powdery mildew. Improve air circulation through pruning.",
                'Cherry_(including_sour)___healthy': "The cherry appears healthy. Monitor for signs of disease and maintain good cultural practices.",
                'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': "Gray leaf spot causes gray lesions on corn leaves, reducing photosynthetic area. Use resistant corn varieties and apply fungicides if needed. Crop rotation and residue management can help.",
                'Corn_(maize)___Common_rust_': "Common rust leads to orange pustules on corn leaves. Use resistant corn varieties and apply fungicides if needed. Timely applications are crucial.",
                'Corn_(maize)___Northern_Leaf_Blight': "Northern leaf blight causes long, elliptical lesions on corn leaves. Use resistant corn varieties and apply fungicides if needed. Crop rotation and residue management can help.",
                'Corn_(maize)___healthy': "The corn appears healthy. Ensure adequate moisture and nutrients.",
                'Grape___Black_rot': "Black rot is a fungal disease causing reddish-brown spots on grape leaves and berries, leading to shriveling and decay. It thrives in warm, humid conditions. **Management:** Remove infected parts promptly, apply fungicides preventatively, especially before and after bloom, and ensure good air circulation through pruning and trellising. Practice sanitation by removing fallen leaves and mummified berries.",
                'Grape___Esca_(Black_Measles)': "Esca, also known as Black Measles, is a complex fungal disease causing leaf discoloration, wood decay, and eventually vine death. It's often associated with stress. **Management:** There's no cure; focus on prevention through good vineyard management, including proper pruning techniques, minimizing stress, and using disease-free planting material. Remove severely infected vines.",
                'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': "Leaf blight causes brown spots on grape leaves, potentially leading to defoliation. **Management:** Apply fungicides preventatively, especially during wet periods. Ensure good air circulation through pruning and trellising. Practice sanitation by removing fallen leaves.",
                'Grape___healthy': "The grape appears healthy. Monitor for signs of disease and maintain good cultural practices.",
                'Orange___Haunglongbing_(Citrus_greening)': "Citrus greening causes yellowing and misshapen fruit. No cure; control psyllid vectors with insecticides and remove infected trees.",
                'Peach___Bacterial_spot': "Bacterial spot causes dark spots on peach leaves and fruit, reducing yield and quality. Use bactericides and plant resistant varieties.",
                'Peach___healthy': "The peach appears healthy. Monitor for signs of disease and maintain good cultural practices.",
                'Pepper,_bell___Bacterial_spot': "Bacterial spot causes lesions on pepper leaves and fruit. Use bactericides and plant resistant varieties.",
                'Pepper,_bell___healthy': "The pepper appears healthy. Ensure proper spacing and irrigation.",
                'Potato Early blight': "Early blight causes dark spots on potato leaves, reducing yield. Use fungicides and plant resistant varieties.",
                'Potato___Late_blight': "Late blight causes water-soaked lesions on potato leaves and tubers, leading to rapid decay. Use fungicides and plant resistant varieties. Monitor weather conditions closely.",
                'Potato___healthy': "The potato appears healthy. Ensure proper hilling and irrigation.",
                'Raspberry___healthy': "The raspberry appears healthy. Ensure proper pruning and support.",
                'Soybean___healthy': "The soybean appears healthy. Monitor for signs of disease and ensure proper irrigation.",
                'Squash___Powdery_mildew': "Powdery mildew causes white, powdery growth on squash leaves and stems. Use fungicides and improve air circulation.",
                'Not Recognised': "The plant disease could not be recognized. Please provide a clearer image.",
                'Strawberry___healthy': "The strawberry appears healthy. Ensure proper mulching and irrigation.",
                'Tomato___Bacterial_spot': "Bacterial spot causes lesions on tomato leaves and fruit. Use bactericides and plant resistant varieties.",
                'Tomato___Early_blight': "Early blight causes dark spots on tomato leaves, reducing yield. Use fungicides and plant resistant varieties.",
                'Tomato___Late_blight': "Late blight causes water-soaked lesions on tomato leaves and fruit, leading to rapid decay. Use fungicides and plant resistant varieties. Monitor weather conditions closely.",
                'Tomato___Leaf_Mold': "Leaf mold causes olive-green to brown spots on tomato leaves. Use fungicides and improve air circulation.",
                'Tomato___Septoria_leaf_spot': "Septoria leaf spot causes small, circular spots on tomato leaves. Use fungicides and practice crop rotation.",
                'Tomato___Spider_mites Two-spotted_spider_mite': "Spider mites cause yellowing and webbing on tomato leaves. Use miticides and ensure proper irrigation.",
                'Tomato___Target_Spot': "Target spot causes concentric rings on tomato leaves and fruit. Use fungicides and practice crop rotation.",
                'Tomato___Tomato_Yellow_Leaf_Curl_Virus': "Yellow leaf curl virus causes yellowing and curling of tomato leaves. Control whitefly vectors with insecticides and remove infected plants.",
                'Tomato___Tomato_mosaic_virus': "Tomato mosaic virus causes mottling and distortion of tomato leaves. No cure; remove infected plants and control aphid vectors.",
                'Tomato___healthy': "The tomato appears healthy. Ensure proper staking and irrigation."
            }

            prediction_label = class_name[result_index]
            diagnosis = diagnosis_info.get(prediction_label, "Diagnosis information not available.")

            st.success(f"Prediction: {prediction_label}\n\nDiagnosis: {diagnosis}")