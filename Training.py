import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical

# Step 1: Load and Prepare the Dataset
X = []  # Features (images)
y = []  # Labels (1 for fire, 0 for non-fire)

# Paths to your dataset folders
fire_directory = "D:\dsa dataset\image_dataset\burn_images"
non_fire_directory = "D:\dsa dataset\image_dataset\not_images"

# Load fire images and assign label 1
for filename in os.listdir(fire_directory):
    if filename.endswith('.jpg'):
        image = cv2.imread(os.path.join(fire_directory, filename))
        # Resize the image to a fixed size (e.g., 128x128 pixels)
        image = cv2.resize(image, (128, 128))
        X.append(image)
        y.append(1)  # 1 represents fire

# Load non-fire images and assign label 0
for filename in os.listdir(non_fire_directory):
    if filename.endswith('.jpg'):
        image = cv2.imread(os.path.join(non_fire_directory, filename))
        # Resize the image to a fixed size (e.g., 128x128 pixels)
        image = cv2.resize(image, (128, 128))
        X.append(image)
        y.append(0)  # 0 represents non-fire

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Normalize pixel values to be between 0 and 1
X = X.astype('float32') / 255.0

# Convert labels to one-hot encoding
y = to_categorical(y, 2)  # 2 is the number of classes (fire and non-fire)

# Step 2: Split the Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Build the CNN Model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='softmax'))  # Output layer with 2 units (fire and non-fire)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 4: Train the Model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Step 5: Evaluate the Model (optional)
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)
