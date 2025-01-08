import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model

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
    from tensorflow import keras
    import tensorflow_hub as hub

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

# Streamlit UI
# Add custom CSS for Dark Theme styling
st.markdown(
    """
    <style>
        .stApp {
            background-color: #121212;  /* Dark background */
            color: #E0E0E0;  /* Light text */
        }
        h1 {
            color: #BB86FC;  /* Accent color for title */
            text-align: center;
            font-weight: bold;
        }
        .stButton button {
            background-color: #03DAC6;  /* Teal button */
            color: black;
            border-radius: 10px;
            font-size: 16px;
            padding: 10px 20px;
        }
        .stButton button:hover {
            background-color: #018786;  /* Darker teal on hover */
        }
        .stSpinner {
            color: #BB86FC !important;
        }
        .result-box {
            background-color: #1E1E1E;
            color: #03DAC6;  /* Teal text for result */
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            text-align: center;
            font-size: 18px;
            font-weight: bold;
        }
        .confidence-box {
            background-color: #1E1E1E;
            color: #FFB74D;  /* Orange for confidence */
            padding: 15px;
            border-radius: 10px;
            margin-top: 10px;
            text-align: center;
            font-size: 18px;
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Header and instructions
st.title("Skin Disease Prediction")
st.write("Upload an image of the skin condition, and let our AI model assist in predicting the disease!")

# Upload image
uploaded_file = st.file_uploader("📂 Upload an image (JPG, JPEG, PNG):", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image with a border
    image = Image.open(uploaded_file)
    st.image(
        image,
        caption="Uploaded Image 📸",
        use_column_width=True,
        output_format="auto",
        clamp=False,
    )

    # Predict button
    if st.button("Predict Skin Condition"):
        with st.spinner("🤖 Analyzing... Please wait."):
            predicted_disease, confidence = predict_disease(image)
            
            # Display Prediction Result
            st.markdown(
                f"""
                <div class="result-box">
                    🔍 Prediction: {predicted_disease.capitalize()}
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Display Confidence Level
            st.markdown(
                f"""
                <div class="confidence-box">
                    🔥 Confidence Level: {confidence * 100:.2f}%
                </div>
                """,
                unsafe_allow_html=True
            )
else:
    st.info("👈 Please upload an image to get started!")
