# 🏥 Skin Disease Detection 🏥

This project is a **Skin Disease Detection Web App** powered by **Streamlit**, **TensorFlow**, and **MobileNet V2**. The application predicts various skin conditions based on an uploaded image, offering real-time predictions with high accuracy.

🔗 **Live at**: [Skin Disease Detection](https://emfqsh5caci5pgjgamukfj.streamlit.app/)

## 🎯 Features
- **Upload Image**: Users can upload skin condition images in .jpg, .jpeg, or .png formats.
- **AI Predictions**: The application predicts one of the following skin conditions:
  - Cellulitis
  - Impetigo
  - Athlete's Foot
  - Nail Fungus
  - Ringworm
  - Cutaneous Larva Migrans
  - Chickenpox
  - Shingles
  - Normal (Healthy skin)
- **Confidence Level**: Displays the model's confidence percentage for the predicted disease.
- **Dark-Themed UI**: Enhanced user experience with a visually appealing dark theme.
- **Real-Time Analysis**: Fast predictions using a lightweight and efficient model architecture.

## 🛠️ Technologies Used
- **Frontend**: Streamlit for an intuitive and interactive web interface.
- **Backend**: TensorFlow and TensorFlow Hub for deep learning model predictions.
- **Pre-Trained Model**: MobileNet V2 fine-tuned for skin disease classification.
- **Image Processing**: Pillow (PIL) and NumPy for preprocessing the input images.

## 📜 Dataset
The model is trained on a dataset of skin disease images sourced from Kaggle.
- The dataset was sourced from: [Skin Disease Dataset on Kaggle](https://www.kaggle.com/datasets/subirbiswas19/skin-disease-dataset)
- The dataset includes the following eight classes of skin diseases:
  - **Cellulitis** (Bacterial Infection)
  - **Impetigo** (Bacterial Infection)
  - **Athlete's Foot** (Fungal Infection)
  - **Nail Fungus** (Fungal Infection)
  - **Ringworm** (Fungal Infection)
  - **Cutaneous Larva Migrans** (Parasitic Infection)
  - **Chickenpox** (Viral Infection)
  - **Shingles** (Viral Infection)
 
  ## 📍 Paper Publication :
  - **ICRAEST-2025 – An international-level conference held on March 21, 2025 at Godavari College of Engineering, Jalgaon, with IEEE technical sponsorship.**[Paper Publication](https://drive.google.com/file/d/1_7kq3iWCUiEbQ4zNNCjnNupQTpo8YGUw/view?usp=sharing)

## 🧠 How It Works
1. **Upload an Image**: Users upload a skin condition image.
2. **Preprocessing**: The image is resized to 224x224 and normalized for prediction.
3. **Prediction**: The pre-trained MobileNet V2 model predicts the disease based on the input image.
4. **Output**: The predicted disease and its confidence level are displayed in a user-friendly interface.

## 🚀 Usage
Follow these steps to use the application locally:

1. **Clone the Repository**  
   Clone the repository to your local machine:  
   `https://github.com/MajorProject-ai/Detect-skin-disease-.git`

2. **Install Required Dependencies**  
   Make sure you have Python 3.8 or later installed. Then, run:  
   `pip install -r requirements.txt`

3. **Add Model Weights**  
   Download the pre-trained model weights (`my_model.weights.h5`) and place them in the project directory.

4. **Run the Application**  
   Run the Streamlit app with:  
   `streamlit run app.py`

## 🎯 Contributor
- **Nishant Kumar (Developer)**
- **Nitish Kumar (Developer)**
- **Rahul Kumar (Developer)**
