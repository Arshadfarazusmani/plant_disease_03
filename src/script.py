import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pymongo import MongoClient
from passlib.hash import pbkdf2_sha256
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


# --- MongoDB Atlas Setup ---
MONGODB_URI = os.environ.get("MONGOURI")
if not MONGODB_URI:
    st.error(
        "MONGODB_URI environment variable not set.  Please set it according to the instructions below."
    )
    st.stop()
DB_NAME = "plant_disease_db"
try:
    client = MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    diseases_collection = db["diseases"]  # Correct collection name.
    users_collection = db["users"] # Make sure you have the users_collection
    images_collection = db["images"]
    predictions_collection = db["predictions"]
except Exception as e:
    st.error(f"Error connecting to MongoDB: {e}")
    st.stop()

# --- Authentication Functions ---
def get_user(username):
    """Retrieves a user from MongoDB by username."""
    try:
        return users_collection.find_one({"username": username})
    except Exception as e:
        st.error(f"Error retrieving user: {e}")
        return None

def add_user(username, password, email):
    """Adds a new user to MongoDB."""
    hashed_password = pbkdf2_sha256.hash(password)
    registration_date = datetime.now()
    try:
        user_data = {
            "username": username,
            "password_hash": hashed_password,
            "email": email,
            "registration_date": registration_date,
        }
        users_collection.insert_one(user_data)
        return True
    except Exception as e:
        st.error(f"Error adding user: {e}")
        return False

def verify_password(password, hashed_password):
    """Verifies a password against its hash."""
    return pbkdf2_sha256.verify(password, hashed_password)


# --- Image and Prediction Functions ---
def save_image(user_id, image_path):
    """Saves image information to the database."""
    upload_datetime = datetime.now()
    try:
        image_data = {
            "user_id": user_id,
            "upload_datetime": upload_datetime,
            "file_path": image_path,
        }
        image_id = images_collection.insert_one(image_data).inserted_id
        return image_id
    except Exception as e:
        st.error(f"Error saving image: {e}")
        return None


def save_prediction(image_id, disease_id, confidence_score):
    """Saves prediction information to the database."""
    prediction_datetime = datetime.now()
    try:
        prediction_data = {
            "image_id": image_id,
            "disease_id": disease_id,
            "prediction_datetime": prediction_datetime,
            "confidence_score": confidence_score,
        }
        predictions_collection.insert_one(prediction_data)
    except Exception as e:
        st.error(f"Error saving prediction: {e}")

def get_disease_id(disease_name):
    """Retrieves disease id from the database using disease name."""
    try:
        disease = diseases_collection.find_one({"disease_name": disease_name})
        if disease:
            return disease["disease_id"]
        else:
            return None
    except Exception as e:
        st.error(f"Error retrieving disease ID: {e}")
        return None




def add_disease(disease_name, symptoms, treatment, image_url=""):
    """Adds a new disease to the database.

    Args:
        disease_name (str): The name of the disease (e.g., "Apple Scab").
        symptoms (list): A list of symptoms (e.g., ["Dark spots on leaves", "Lesions on fruit"]).
        treatment (str):  A description of the treatment.
        image_url (str, optional): URL of an image representing the disease. Defaults to "".
    """
    try:
        # Check if the disease already exists
        if diseases_collection.find_one({"disease_name": disease_name}):
            st.warning(f"Disease '{disease_name}' already exists in the database.")
            return False

        disease_data = {
            "disease_name": disease_name,
            "symptoms": symptoms,
            "treatment": treatment,
            "image_url": image_url,  # Added image URL
        }
        diseases_collection.insert_one(disease_data)
        st.success(f"Disease '{disease_name}' added successfully.")
        return True
    except Exception as e:
        st.error(f"Error adding disease '{disease_name}': {e}")
        return False

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


def main():
    st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿", layout="wide")
    if 'user' not in st.session_state:
        st.session_state.user = None

    # Custom CSS (Green Theme)
    st.markdown(
        """
    <style>
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
        color: #1e5638; /* Dark green */
    }
    .sidebar .sidebar-content {
        background-color: #f0f4f3; /* Very light green */
        border-right: 1px solid #d4edda; /* Light green border */
    }
    .stButton > button {
        background-color: #4CAF50; /* Green */
        color: white;
        border: none;
        border-radius: 0.3rem;
        padding: 0.75rem 1.5rem;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.15);
        transition: background-color 0.3s ease, transform 0.2s ease;
        width: 100%;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .stButton > button:hover {
        background-color: #45a049; /* Darker green hover */
        transform: translateY(-1px);
        box-shadow: 0 3px 7px rgba(0, 0, 0, 0.2);
    }
    .stSuccess {
        background-color: #d4edda; /* Lightest green */
        color: #155724;
        padding: 1rem;
        border-radius: 0.3rem;
        margin-bottom: 1rem;
        border: 1px solid #c3e6cb; /* Light green border */
        font-size: 1rem;
    }
    .stError {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.3rem;
        margin-bottom: 1rem;
        border: 1px solid #f5c6cb;
        font-size: 1rem;
    }
    .stImage {
        border-radius: 0.5rem;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )



    # Sidebar
    with st.sidebar:
        st.title("Navigation")
        if st.session_state.user:
            st.write(f"Welcome, {st.session_state.user['username']}!")
            app_mode = st.radio("Select Page", ["Home", "Disease Recognition", "About", "Logout", "Profile", "Add Disease"])
        else:
            app_mode = st.radio("Select Page", ["Home", "Login", "Signup", "Disease Recognition", "About", "Add Disease"]) # Add "Add Disease" here

    if app_mode == "Add Disease":
        st.title("Add Diseases to Database")

        # Text input for disease name
        disease_name = st.text_input("Disease Name:")
        # Text area for symptoms, allows multiple lines
        symptoms_input = st.text_area("Symptoms (comma-separated):")
        # Text area for treatment
        treatment = st.text_area("Treatment:")
        # Text input for image URL
        image_url = st.text_input("Image URL (optional):")

        if st.button("Add Disease"):
            # Convert comma-separated symptoms to a list
            symptoms = [s.strip() for s in symptoms_input.split(",")] if symptoms_input else []
            if disease_name and symptoms and treatment:  # Basic validation
                add_disease(disease_name, symptoms, treatment, image_url)
            else:
                st.error("Please provide disease name, symptoms, and treatment.")

        # Add this part to add multiple diseases from the list
        disease_list = [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
            'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
            'Cherry_(including_sour)___healthy',
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
            'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
            'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
            'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
            'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
            'Potato Early blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy',
            'Soybean___healthy', 'Squash___Powdery_mildew', 'Not Recognised', 'Strawberry___healthy',
            'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
            'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
            'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
        ]

        if st.button("Add Multiple Diseases"):
            for disease_name in disease_list:
                #  You would need to provide default or empty values for symptoms and treatment.
                #  Since you don't have this information in your list
                add_disease(disease_name, symptoms=[], treatment="") #Added empty lists
            st.success("All diseases from the list have been added (if they didn't already exist).")


    # The rest of your main function (Home, About, Login, Signup, Logout, Profile, Disease Recognition)
    # ... (Keep the existing code for these pages)
    # Home Page
    elif app_mode == "Home":
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image("images/home_page.jpeg", use_container_width=True, output_format="PNG")
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
                1.  **Upload Image:** Navigate to the 'Disease Recognition' page and upload a plant image.
                2.  **Analysis:** Our system processes the image using deep learning.
                3.  **Results:** Receive instant results with disease identification.
                """
            )
            st.markdown(
                """
                ### Why Choose Us:
                -   **Accuracy:** Advanced AI for reliable detection.
                -   **Ease of Use:** User-friendly interface.
                -   **Efficiency:** Fast results for timely action.
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
            -   **Training Set:** 70,295 images
            -   **Test Set:** 33 images
            -   **Validation Set:** 17,572 images
            """
        )

     # Login Page
    elif app_mode == "Login":
        st.title("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Log In"):
            user = get_user(username)
            if user and verify_password(password, user['password_hash']):
                st.session_state.user = {
                    "user_id": str(user["_id"]),
                    "username": user["username"],
                    "email": user.get("email", ""),
                    "registration_date": user.get("registration_date"),
                }
                st.success("Logged in successfully!")
                st.experimental_rerun()
            else:
                st.error("Invalid username or password.")

    # Signup Page
    elif app_mode == "Signup":
        st.title("Sign Up")
        new_username = st.text_input("Username")
        new_password = st.text_input("Password", type="password")
        new_email = st.text_input("Email")
        if st.button("Sign Up"):
            if add_user(new_username, new_password, new_email):
                st.success("Signed up successfully! Please log in.")
            else:
                st.error("Username already exists. Please choose a different one.")

    # Logout
    elif app_mode == "Logout":
        st.session_state.user = None
        st.success("Logged out!")
        st.experimental_rerun()

    # Profile Page
    elif app_mode == "Profile":
        if not st.session_state.user:
            st.warning("Please log in to view your profile.")
        else:
            st.title("Your Profile")
            user = st.session_state.user
            st.write(f"**Username:** {user['username']}")
            st.write(f"**Email:** {user['email']}")
            st.write(f"**Registration Date:** {user['registration_date'].strftime('%Y-%m-%d %H:%M:%S')}") #format the date

            # Display user's image uploads and predictions
            st.subheader("Your Uploaded Images and Predictions")
            try:
                user_images = images_collection.find({"user_id": user['user_id']})
                for image in user_images:
                    st.image(image['file_path'], caption=f"Uploaded: {image['upload_datetime'].strftime('%Y-%m-%d %H:%M:%S')}", use_column_width=True)
                    prediction = predictions_collection.find_one({"image_id": image['_id']}) # changed image_id
                    if prediction:
                        disease = diseases_collection.find_one({"disease_id": prediction['disease_id']})
                        if disease:
                            st.write(f"  **Predicted Disease:** {disease['disease_name']}")
                            st.write(f"  **Confidence:** {prediction['confidence_score']:.2f}")
                        else:
                            st.write("  **Predicted Disease:** Not found")
                    else:
                        st.write("  **Prediction:** No prediction available")
            except Exception as e:
                st.error(f"Error retrieving user data: {e}")
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


if __name__ == "__main__":
    main()

