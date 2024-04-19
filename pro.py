#code for displaying 5 images from both datasets
import os
from PIL import Image
import matplotlib.pyplot as plt

def display_images(folder_path, num_images=5):
    # Get the list of files in the folder
    files = os.listdir(folder_path)

    # Display the first num_images images
    for i in range(min(num_images, len(files))):
        image_path = os.path.join(folder_path, files[i])
        image = Image.open(image_path)

        # Display the image
        plt.subplot(1, num_images, i+1)
        plt.imshow(image)
        plt.axis('off')

    plt.show()

# Specify the paths to your folders
main_folder = "/content/drive/MyDrive/Final Project/Dataset"
benign_folder = os.path.join(main_folder, "Benign")
pro_folder = os.path.join(main_folder, "Pro")

# Display first 5 images from the 'benign' folder
print("Displaying first 5 images from 'benign' folder:")
display_images(benign_folder)

# Display first 5 images from the 'pro' folder
print("Displaying first 5 images from 'pro' folder:")
display_images(pro_folder)



#split data into train and test
import os
import shutil
from sklearn.model_selection import train_test_split

# Define your main dataset directory
main_dataset_dir = "/content/drive/MyDrive/Final Project/Dataset"

# Define the directories for benign and pro images
benign_dir = os.path.join(main_dataset_dir, "Benign")
pro_dir = os.path.join(main_dataset_dir, "Pro")

# Define the directories for training and test sets
train_dir = "/content/drive/MyDrive/Final Project/Train"
test_dir = "/content/drive/MyDrive/Final Project/Test"

# Create the training and test directories
os.makedirs(os.path.join(train_dir, "Benign"), exist_ok=True)
os.makedirs(os.path.join(train_dir, "Pro"), exist_ok=True)
os.makedirs(os.path.join(test_dir, "Benign"), exist_ok=True)
os.makedirs(os.path.join(test_dir, "Pro"), exist_ok=True)

# Function to split the dataset into training and test sets
def split_dataset(src_dir, train_dst, test_dst, test_size=0.2, random_state=42):
    # Get the list of all image files in the source directory
    all_images = [f for f in os.listdir(src_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # Split the images into training and test sets
    train_images, test_images = train_test_split(all_images, test_size=test_size, random_state=random_state)

    # Copy images to the training set
    for img in train_images:
        src_path = os.path.join(src_dir, img)
        dst_path = os.path.join(train_dst, img)
        shutil.copy(src_path, dst_path)

    # Copy images to the test set
    for img in test_images:
        src_path = os.path.join(src_dir, img)
        dst_path = os.path.join(test_dst, img)
        shutil.copy(src_path, dst_path)

# Split the dataset for benign images
split_dataset(benign_dir, os.path.join(train_dir, "Benign"), os.path.join(test_dir, "Benign"))

# Split the dataset for pro images
split_dataset(pro_dir, os.path.join(train_dir, "Pro"), os.path.join(test_dir, "Pro"))

#Augmentation




#code of data augmentation
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import os

# Define the path to your original training dataset
original_train_dir = "/content/drive/MyDrive/Final Project/Train"

# Define the path to your augmented training dataset
augmented_train_dir = "/content/drive/MyDrive/Final Project/Augmentation"

# Create an ImageDataGenerator with augmentation parameters for the 'Benign' subfolder
datagen_benign = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest"
)

# Create an ImageDataGenerator with augmentation parameters for the 'Pro' subfolder
datagen_pro = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest"
)

# Iterate through each class (subfolder) in the original training dataset
for class_name in os.listdir(original_train_dir):
    class_path = os.path.join(original_train_dir, class_name)

    # Create separate folder for augmented images in each class
    augmented_class_path = os.path.join(augmented_train_dir, class_name)
    os.makedirs(augmented_class_path, exist_ok=True)

    # Set the appropriate datagen for each class
    if class_name == 'Benign':
        current_datagen = datagen_benign
        augment_count = 3
    elif class_name == 'Pro':
        current_datagen = datagen_pro
        augment_count = 3

    # Iterate through images in the original class folder
    for filename in os.listdir(class_path):
        img_path = os.path.join(class_path, filename)
        img = load_img(img_path)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        # Generate augmented images and save to the new folder
        i = 0
        for batch in current_datagen.flow(x, batch_size=1, save_to_dir=augmented_class_path, save_prefix='aug', save_format='jpeg'):
            i += 1
            if i >= augment_count:
                break  # This ensures that we generate the specified number of augmented images for each original image
