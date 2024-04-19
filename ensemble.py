import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the pre-trained models
inception_model = load_model("/content/drive/MyDrive/Updated3/inceptionv3_trained_model.h5")
xception_model = load_model("/content/drive/MyDrive/Upadated1/xception_trained_model.h5")
mobilenet_model = load_model("/content/drive/MyDrive/Normal/mobilenet_trained_model.h5")

# Function to preprocess input image for Inception
def preprocess_image_inception(img_path):
    img = Image.open(img_path)
    img = img.resize((299, 299))  # Resize to match Inception input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to preprocess input image for Xception
def preprocess_image_xception(img_path):
    img = Image.open(img_path)
    img = img.resize((299, 299))  # Resize to match Xception input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to preprocess input image for MobileNet
def preprocess_image_mobilenet(img_path):
    img = Image.open(img_path)
    img = img.resize((224, 224))  # Resize to match MobileNet input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Assuming your main directory containing the test dataset is "/content/drive/MyDrive/Test1/"
main_directory = "/content/drive/MyDrive/Test1"

# Get the list of subfolders (class labels)
class_folders = os.listdir(main_directory)

# Initialize variables to keep track of correct predictions
correct_predictions = 0
total_images = 0

# Iterate through each class folder
for class_folder in class_folders:
    class_path = os.path.join(main_directory, class_folder)

    # Get the list of image files in the class folder
    image_files = os.listdir(class_path)

    # Iterate through each image file
    for image_file in image_files:
        # Construct the full path to the image file
        image_path = os.path.join(class_path, image_file)

        # Preprocess the input images for each model
        inception_img = preprocess_image_inception(image_path)
        xception_img = preprocess_image_xception(image_path)
        mobilenet_img = preprocess_image_mobilenet(image_path)

        # Predict the class of the input images using each model
        inception_pred = inception_model.predict(inception_img)
        xception_pred = xception_model.predict(xception_img)
        mobilenet_pred = mobilenet_model.predict(mobilenet_img)

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
        final_prediction = "Benign" if benign_count > malignant_count else "Pro"

        # Compare with the ground truth label (based on the class folder name)
        if final_prediction == class_folder:  # Assuming class folder names are strings
            correct_predictions += 1

        total_images += 1

# Calculate accuracy
accuracy = correct_predictions / total_images
print("Accuracy:", accuracy)
