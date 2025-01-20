import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
from tensorflow import keras
import tensorflow_hub as hub
import cohere  # Cohere import

# Set page configuration
st.set_page_config(page_title="Skin Disease Detection", layout="wide")

# Apply custom CSS for styling
st.markdown(
    """
    <style>
    body {
        background-color: #ffffff;  /* Changed background color to white */
    }

    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 80px;
        text-align: center;
        font-size: 18px;
        border-radius: 5px;
        transition: 0.3s;
        width: 100%;
    }
    /* Sidebar box styling */
    .sidebar .sidebar-content {
        background-color: #d6f5e9;
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 10px;
        margin: 10px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stFileUploader>label {
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load the trained model weights
model = None
disease_train_label_dic = {
    'cellulitis': 0,
    'impetigo': 1,
    'athlete-foot': 2,
    'nail-fungus': 3,
    'ringworm': 4,
    'cutaneous-larva-migrans': 5,
    'chickenpox': 6,
    'shingles': 7,
    'normal': 8
}

# Initialize and load weights
def load_model_weights():
    global model
    
    feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
    feature_extractor_layer = hub.KerasLayer(feature_extractor_url, input_shape=(224, 224, 3), trainable=False)

    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(224, 224, 3)),
        keras.layers.Lambda(lambda x: feature_extractor_layer(x)),
        keras.layers.Dense(len(disease_train_label_dic), activation='softmax')
    ])
    model.load_weights("my_model.weights.h5")

load_model_weights()

# Preprocess the image
def preprocess_image(image: Image.Image):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0  # Normalize pixel values
    return np.expand_dims(img_array, axis=0)

# Predict the disease
def predict_disease(image: Image.Image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction)
    predicted_disease = next((disease for disease, label in disease_train_label_dic.items() if label == predicted_class), "Unknown")
    return predicted_disease, prediction[0][predicted_class]

# Fetch treatment suggestion using Cohere API
def fetch_treatment(disease_name):
    # API_KEY = st.secrets["MAJOR_API_KEY"]  # Fetch the API key from secrets
    API_KEY = "5yXaUpp4tkDx2hGwX4VhjzKavVFBbnMi80YTAupB" # Fetch the API key from secrets
    co = cohere.Client(API_KEY)  # Initialize Cohere client

    try:
        response = co.generate(
            model='command',
            prompt=f'You are a medical specialist. Suggest treatment for {disease_name}.',
            max_tokens=1024,
            temperature=0.750
        )
        treatment = response.generations[0].text
        return treatment
    except Exception as e:
        return f"Error fetching treatment: {str(e)}"

# Sidebar menu with clickable boxes
st.sidebar.title("Menu")
st.sidebar.markdown("---")  # Divider for better look
menu_options = ["Home", "Video", "About", "Contact"]
selected_option = "Home"  # Default to Home page

for option in menu_options:
    if st.sidebar.button(option):
        selected_option = option

# Home Page
if selected_option == "Home":
    st.markdown(
        """
        <h1 style="text-align: center; font-weight: bold; color: #060606;">Skin Disease Detection</h1>
        <h3 style="text-align: center; color: #060606;">Automated Diagnosis of Skin Diseases with Image Recognition 🩺💻</h3>
        <p style="text-align: center;">Upload your skin disease image to get a diagnosis</p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <p style="color: red; font-size: 18px; text-align: center;">
        Please note that although our model achieves a 92% accuracy rate, its predictions should be considered with a limited guarantee. Determining the precise type of skin lesion should be done by a qualified doctor for an accurate diagnosis.
        </p>
        """,
        unsafe_allow_html=True,
    )

    if st.checkbox("I understand and accept", key="home_checkbox"):
        uploaded_file = st.file_uploader(
            "Drag and drop file here or browse files",
            type=["png", "jpg", "jpeg"],
            help="Limit 200MB per file • PNG, JPG, JPEG",
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded File", width=300)

            if st.button("Predict", key="home_predict"):
                with st.spinner("Processing....."):
                    predicted_disease, confidence = predict_disease(image)

                    # Display Prediction Result
                    st.markdown(
                        f"""
                        <div style="background-color: #1E1E1E; color: #ffffff; padding: 20px; border-radius: 7px; margin-top: 20px; text-align: center; font-size: 18px; font-weight: bold;">
                            Expected Result: {predicted_disease.capitalize()}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Display Confidence Level
                    st.markdown(
                        f"""
                        <div style="background-color: #1E1E1E; color: #ffffff; padding: 15px; border-radius: 7px; margin-top: 10px; text-align: center; font-size: 18px; font-weight: bold;">
                            Assurance Level: {confidence * 100:.2f}%
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Fetch and display treatment suggestion
                    treatment = fetch_treatment(predicted_disease)
                    st.markdown(
                        f"""
                        <div style="background-color: #1E1E1E; color: #ffffff; padding: 20px; border-radius: 7px; margin-top: 20px; text-align: center; font-size: 18px; font-weight: bold;">
                            Suggested Treatment: {treatment}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        else:
            st.button("Predict", disabled=True)
    else:
        st.info("Please accept the terms to proceed.")

# Demo Video Page
elif selected_option == "Video":
    st.markdown("<h2 style='text-align: center; color: #056839;'>Demo Video</h2>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Upload your video</h3>", unsafe_allow_html=True)
  #  st.video("#")  # Replace with your demo video link
    

# About Page
elif selected_option == "About":
    st.markdown("<h2 style='text-align: center; color: #056839;'>About</h2>", unsafe_allow_html=True)

    # About the Developers
    st.subheader("About the Developers:")
    st.write(
        "Skin Disease Detection is developed by Nishant Kumar, Rahul Kumar, and Nithish Kumar, who are currently pursuing a degree in Computer Science Engineering at Dr. MGR Educational and Research Institute, Maduravoyal, Chennai 600095, Tamil Nadu, India."

    )
    st.write(
        "The project has been carried out under the expert guidance and supervision of our esteemed advisors: Dr. R. Sudhakar, Dr. J. Jayaprakash, and Mrs. Chinchu Nair. Their invaluable insights and mentorship have played a crucial role in the successful development of this project."
    )

    # IEEE Publication section
    st.subheader("Our paper Publication")
    st.write(
        # "Our project also led to the publication of a paper in IEEE, detailing our findings and innovations in skin disease diagnosis using AI. "
        "data "
        "[Link](#)"
    )

    # About the Project section
    st.subheader("About the Project:")
    st.write(
        # "Teledermatologis AI is the fruitful result of our final project in our software engineering degree. This initiative was motivated by the escalating demand for advanced tools to accurately diagnose skin conditions in family clinics."
    )
    st.write(
        # "Recognizing that precise diagnosis is fundamental for effective treatment, our goal was to create an advanced system capable of automatically identifying different types of skin lesions. "
        # "Our focus centered on six key skin conditions: Acne, Basal cell carcinoma, Eczema, Seborrheic keratosis, Milia, and Rosacea. "
        # "These conditions are frequently encountered in clinical settings, and accurate identification can significantly enhance patient care outcomes."
        "data"
    )

    # Technical Details section
    st.subheader("Technical Details:")
    st.write(
        # "In order to achieve this, we used machine learning and image processing techniques. This led to the creation of an intelligent system that achieves an impressive 92% accuracy rate on our self-curated dataset. "
        # "The dataset, comprised of 1,700 images, was divided into 75% for training, 10% for validation, and 15% for testing. "
        # "This division allowed us to refine our model to be robust and reliable, providing a high degree of confidence in its diagnostic capabilities."
        "Data"
    )

    # Closing statement
    st.write(
        # "Teledermatologis AI is the successful end product of our software engineering degree project. It epitomizes our dedication to learning new technologies, achieving our goals, and pursuing excellence. "
        # "With the impressive accuracy of the system, we hope to help as many people as possible. \nMay we all enjoy good health!"
        
    )


# Contact Us Page
elif selected_option == "Contact":
    st.markdown("<h2 style='text-align: center; color: #000000;'>Contact Us</h2>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align: center; color: #333;">
            <p>We would love to hear from you!</p>
            <p>Feel free to reach out to us with any inquiries or feedback.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Create a table in markdown format
    st.markdown(
        """
        <table style="margin-left: auto; margin-right: auto; border-collapse: collapse; width: 80%; text-align: left; border: 1px solid #ddd; background-color: #f9f9f9;">
            <thead style="background-color: #e3f2fd;">
                <tr>
                    <th style="padding: 10px; border: 1px solid #ddd;">Name</th>
                    <th style="padding: 10px; border: 1px solid #ddd;">LinkedIn Profile</th>
                    <th style="padding: 10px; border: 1px solid #ddd;">Mail</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd;">Nishant kumar</td>
                    <td style="padding: 10px; border: 1px solid #ddd;">
                        <a href="https://www.linkedin.com/in/nishant-kumar-247899240/" style="color: #007bff; text-decoration: none;">Nishant kumar Profile</a>
                    </td>
                    <td style="padding: 10px; border: 1px solid #ddd;">rajhanshsingh99882@gmail.com</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd;">Rahul kumar</td>
                    <td style="padding: 10px; border: 1px solid #ddd;">
                        <a href="https://www.linkedin.com/in/rahulrajsharma1/" style="color: #007bff; text-decoration: none;">Rahul kumar Profile</a>
                    </td>
                    <td style="padding: 10px; border: 1px solid #ddd;">rahulrajsharma512@gmail.com</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd;">Nitish kumar</td>
                    <td style="padding: 10px; border: 1px solid #ddd;">
                        <a href="https://www.linkedin.com/in/nitish-kumar-676402241/" style="color: #007bff; text-decoration: none;">Nitish kumar Profile</a>
                    </td>
                    <td style="padding: 10px; border: 1px solid #ddd;">nitishhhp57@gmail.com</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd;">Dr. Name </td>
                    <td style="padding: 10px; border: 1px solid #ddd;">
                        <a href="#" style="color: #007bff; text-decoration: none;"> Dr. Profile</a>
                    </td>
                    <td style="padding: 10px; border: 1px solid #ddd;">Dr@gmail.com</td>
                </tr>
            </tbody>
        </table>
        """,
        unsafe_allow_html=True
    )

# requirenment.txt file 

# streamlit
# pillow
# numpy
# tensorflow
# tensorflow-hub
# keras
