#XCEPTION
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report

# Load pre-trained Xception model
base_model = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

# Add custom classification layers on top of Xception
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

# Combine base model and custom layers into a new model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the base model (optional)
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Print model summary
# model.summary()

# Define paths to your preprocessed train and test data directories
train_data_dir = '/content/drive/MyDrive/Dataset1'
test_data_dir = '/content/drive/MyDrive/Test1'

# Set batch size
batch_size = 32

# Set up data generator for training data
train_datagen = ImageDataGenerator()

# Load and preprocess training data
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(299, 299),
    batch_size=batch_size,
    class_mode='binary')

# Load and preprocess test data
test_generator = train_datagen.flow_from_directory(
    test_data_dir,
    target_size=(299, 299),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False)

# Train the model
model.fit(train_generator, epochs=20)

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
print(classification_report(y_true, y_pred_binary))

from sklearn.metrics import confusion_matrix

# Generate confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_binary)
print("Confusion Matrix:")
print(conf_matrix)

model.save("/content/drive/MyDrive/Upadated1/xception_trained_model.h5")
