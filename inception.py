import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from sklearn.metrics import classification_report, confusion_matrix

# Load pre-trained InceptionV3 model
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

# Add custom classification layers on top of InceptionV3
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.5)(x)  # Add dropout layer
x = layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)  # Add L2 regularization
predictions = layers.Dense(1, activation='sigmoid')(x)

# Combine base model and custom layers into a new model
model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the base model (optional)
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(lr=0.01), loss='binary_crossentropy', metrics=['accuracy'])

# Define paths to your preprocessed train and test data directories
train_data_dir = '/content/drive/MyDrive/Dataset1'
test_data_dir = '/content/drive/MyDrive/Test1'

# Set batch size
batch_size = 32

# Set up data generator for training data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Set up data generator for test data (only rescaling)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load and preprocess training data
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(299, 299),
    batch_size=batch_size,
    class_mode='binary'
)

# Load and preprocess test data
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(299, 299),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=test_generator,
    callbacks=[early_stopping]
)

# Evaluate the model
loss, accuracy = model.evaluate(test_generator)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Generate predictions
y_pred = model.predict(test_generator)
y_pred_binary = (y_pred > 0.5).astype(int)

# Get true labels
y_true = test_generator.classes

# Generate classification report
print("Classification Report:")
print(classification_report(y_true, y_pred_binary))

# Generate confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_binary)
print("Confusion Matrix:")
print(conf_matrix)

# Save the trained model
model.save("/content/drive/MyDrive/Updated3/inceptionv3_trained_model.h5")
