import os
import numpy as np
from tensorflow.keras.applications import Xception
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D

# Define the path to the main folder containing subfolders for each class
main_folder = "/content/drive/MyDrive/Dataset1"

# Define the path to the folder to store the extracted features
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

# Function to recursively traverse directories and extract features
def traverse_and_extract(folder_path):
    all_features = []
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(".jpg") or filename.endswith(".png"):
                img_path = os.path.join(root, filename)
                features = extract_features(img_path)
                all_features.append(features)
    all_features = np.concatenate(all_features, axis=0)
    np.save(os.path.join(feature_folder, "all_features.npy"), all_features)

# Recursively traverse directories and extract features
traverse_and_extract(main_folder)
