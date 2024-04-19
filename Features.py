import os
import numpy as np
from tensorflow.keras.applications import Xception
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D

# Define the path to the folder containing the saved features file
feature_folder = "/content/drive/MyDrive/Fea"

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
    
    # Calculate the maximum possible distance between feature vectors
    max_distance = np.sqrt(np.sum(np.square(input_features)) + np.sum(np.square(dataset_features), axis=1))
    
    # Calculate the threshold distance based on the specified percentage
    threshold_distance = max_distance * threshold_percentage
    
    # Check if a match is found based on the threshold distance
    if min_distance < 9.5:
        return min_distance
    else:
        return None

# Example of using the match_features function with an input image
input_image_path = "/content/drive/MyDrive/WhatsApp Image 2024-04-05 at 12.33.07 PM.jpeg"
input_features = extract_features(input_image_path)

# Specify the threshold percentage (e.g., 50%)
threshold_percentage = 0.8

min_distance = match_features(input_features, dataset_features, threshold_percentage)

if min_distance is not None:
    print("Input image matches with dataset features. Minimum distance:", min_distance)
else:
    print("No matching feature found in the dataset.")
