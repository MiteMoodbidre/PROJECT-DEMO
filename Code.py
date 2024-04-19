
from tensorflow.keras.models import load_model
import os
from PIL import Image
import numpy as np
import streamlit as st
from tensorflow.keras.applications import Xception
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D

# Define the path to the folder containing the saved features file
feature_folder = "Fea"

# Initialize Xception model with pre-trained weights from ImageNet
base_model = Xception(weights='imagenet', include_top=False)

# Add a global average pooling layer on top of the base model
x = GlobalAveragePooling2D()(base_model.output)

# Define a new model with the custom global average pooling layer
feature_extraction_model = Model(inputs=base_model.input, outputs=x)

# Function to extract features from an image
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(299, 299))  # Xception requires input size of (299, 299)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = feature_extraction_model.predict(x)
    return features

# Load the saved features from the dataset
dataset_features = np.load(os.path.join(feature_folder, "all_features.npy"))

# Function to compute similarity between two feature vectors
def compute_similarity(input_features, dataset_features):
    # Compute Euclidean distance between input features and dataset features
    distances = np.linalg.norm(input_features - dataset_features, axis=1)
    return distances

# Function to check if input image matches any of the stored features
def match_features(input_features, dataset_features, threshold_percentage):
    distances = compute_similarity(input_features, dataset_features)
    min_distance = np.min(distances)
    
    print(min_distance)
    
    # Calculate the maximum possible distance between feature vectors
    max_distance = np.sqrt(np.sum(np.square(input_features)) + np.sum(np.square(dataset_features), axis=1))
    
    # Calculate the threshold distance based on the specified percentage
    threshold_distance = max_distance * threshold_percentage
    
    # Check if a match is found based on the threshold distance
    if min_distance <= 9.799:
        return min_distance
    else:
        return None

# Load the models outside the prediction block
inception_model = load_model("inceptionv3_trained_model.h5")
xception_model = load_model("xception_trained_model.h5")
mobilenet_model = load_model("mobilenet_trained_model.h5")


def preprocess_image(image, model_name):
    target_size = (299, 299) if model_name == 'inception' or model_name == 'xception' else (224, 224)
    image = image.resize(target_size)
    image = np.expand_dims(image, axis=0)
    return image

# Title and custom background
st.title("Bone Marrow Cancer Detection")
image_below_heading = Image.open('bg.jpg')  # Replace 'bg.jpg' with the actual image path
st.image(image_below_heading, caption='Dr. Smith', use_column_width=True)

# Description of the project
st.markdown("""
    This project aims to detect bone marrow cancer using deep learning models. Please upload an image below to make predictions.
""")

# Display image and checkbox
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg"])

# Show prediction message and precautionary methods only when image is uploaded and matches
if uploaded_file is not None:
    # Load and preprocess the uploaded image
    img = image.load_img(uploaded_file, target_size=(299, 299))  # Xception requires input size of (299, 299)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Extract features from the uploaded image
    input_features = extract_features(uploaded_file)

    # Specify the threshold percentage (e.g., 50%)
    threshold_percentage = 0.8

    # Check if input image matches any of the stored features
    min_distance = match_features(input_features, dataset_features, threshold_percentage)

    if min_distance is not None:
        #st.success("Input image matches with dataset features. Minimum distance: {:.2f}".format(min_distance))
        
        # Preprocess the image for each model\
            
         # Resize for MobileNet
        inception_input = preprocess_image(img, 'inception')
        xception_input = preprocess_image(img, 'xception')
        mobilenet_input = preprocess_image(img, 'mobilenet')

        # Predictions
        inception_pred = inception_model.predict(inception_input)
        print(inception_pred)
        xception_pred = xception_model.predict(xception_input)
        #print(xception_pred)
        mobilenet_pred = mobilenet_model.predict(mobilenet_input)
        #print(mobilenet_pred)

        # Initialize counts for benign and malignant predictions
        benign_count = 0
        malignant_count = 0

        # Determine the predicted class for each model
        if inception_pred[0][0] < 0.5:
            benign_count += 1
        else:
            malignant_count += 1

        if xception_pred[0][0] < 0.5:
            benign_count += 1
        else:
            malignant_count += 1

        if mobilenet_pred[0][0] < 0.5:
            benign_count += 1
        else:
            malignant_count += 1

        # Predict the final class based on the counts
        prediction_message = ""
        if benign_count > malignant_count:
            prediction_message = "Predicted Class: Benign"
            suggestion = "Based on the prediction, it suggests a lower likelihood of cancer. Regular check-ups are still recommended."
            st.markdown(f'<p class="prediction-message benign-message">{prediction_message}</p>', unsafe_allow_html=True)
            st.write(suggestion)
            
            # Display precautionary methods for benign prediction
            st.markdown("<h2>Precautionary Methods</h2>", unsafe_allow_html=True)
            st.write(
                """
                - Follow up with regular medical check-ups to monitor any changes.
                - Maintain a healthy lifestyle with a balanced diet and regular exercise.
                - Stay informed about potential risk factors for bone marrow cancer and take preventive measures as advised by healthcare professionals.
                - Avoid exposure to harmful chemicals or substances that may increase cancer risk.
                """
            )

        else:
            prediction_message = "Predicted Class: Malignant"
            suggestion = "Based on the prediction, it suggests a higher likelihood of cancer. Immediate consultation with a healthcare professional is recommended."
            st.markdown(f'<p class="prediction-message malignant-message">{prediction_message}</p>', unsafe_allow_html=True)
            st.write(suggestion)

            # Display precautionary methods for malignant prediction
            st.markdown("<h2>Precautionary Methods</h2>", unsafe_allow_html=True)
            st.write(
                """
                - Seek immediate consultation with a healthcare professional specializing in oncology.
                - Follow their recommendations for further diagnostic tests and treatment options.
                - Maintain a positive mindset and seek support from family, friends, or support groups.
                - Make lifestyle changes such as quitting smoking, reducing alcohol consumption, and adopting a healthy diet to support overall health during treatment.
                """
        
            )

    else:
        st.error("No matching feature found in the dataset. Prediction stopped.")
