import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam


# Directory containing the images
image_dir = '/Users/andreshofmann/Desktop/Studies/Uol/7t/FP/Data/HAM/ham_images'

# Load the dataset
data = pd.read_csv('/Users/andreshofmann/Desktop/Studies/Uol/7t/FP/Data/HAM/HAM10000_metadata.csv')

# Image size (for resizing)
image_size = (128, 128)

# Function to load and preprocess images using OpenCV
def load_and_preprocess_images_cv(df, image_dir, image_size):
    images = []
    labels = []
    missing_images = []
    for idx, row in df.iterrows():
        img_path = os.path.join(image_dir, f"{row['image_id']}.jpg")
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.resize(img, image_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            labels.append(row['dx'])
        else:
            missing_images.append(img_path)
    images = np.array(images, dtype='float32') / 255.0  # Normalize images
    return images, labels, missing_images

# Load and preprocess the images
images, labels, missing_images = load_and_preprocess_images_cv(data, image_dir, image_size)

# Check if there are missing images
if missing_images:
    print(f"Missing images: {len(missing_images)}")
    for img in missing_images[:10]:  # Display up to 10 missing images for reference
        print(img)

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, encoded_labels, test_size=0.2, random_state=42)

# Display the shape of the datasets
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Testing labels shape: {y_test.shape}")

# Define the neural network architecture
def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='tanh', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='tanh'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='tanh'),
        MaxPooling2D((2, 2)),
        Conv2D(256, (3, 3), activation='tanh'),
        MaxPooling2D((2, 2)),
        Conv2D(512, (3, 3), activation='tanh'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    return model

# Get the input shape and number of classes
input_shape = X_train.shape[1:]
num_classes = len(np.unique(y_train))

# Create the model
model = create_cnn_model(input_shape, num_classes)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=40, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# Plot training history
import matplotlib.pyplot as plt

# Accuracy plot
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')

plt.show()
