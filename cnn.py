import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
import winsound

# Step 1: Load and Prepare the Dataset
X = []  # Features (images)
y = []  # Labels (1 for fire, 0 for non-fire)

# Paths to your dataset folders
fire_directory = r"D:\dsa dataset\image_dataset\burn_images"
non_fire_directory = r"D:\dsa dataset\image_dataset\not_images"

# Load fire images and assign label 1
for filename in os.listdir(fire_directory):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image = cv2.imread(os.path.join(fire_directory, filename))
        # Resize the image to a fixed size (e.g., 128x128 pixels)
        image = cv2.resize(image, (128, 128))
        X.append(image)
        y.append(1)  # 1 represents fire
        #print(f"Loaded fire image: {filename}")  # Print the loaded image filename

# Load non-fire images and assign label 0
for filename in os.listdir(non_fire_directory):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image = cv2.imread(os.path.join(non_fire_directory, filename))
        if image is not None:
            # Resize the image to a fixed size (e.g., 128x128 pixels)
            image = cv2.resize(image, (128, 128))
            X.append(image)
            y.append(0)  # 0 represents non-fire
        else:
            print("Failed to load image:", os.path.join(non_fire_directory, filename))
    else:
        print("Invalid file format:", filename)

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

print("Number of samples in X:", len(X))
print("Number of samples in y:", len(y))

# Normalize pixel values to be between 0 and 1
X = X.astype('float32') / 255.0

# Convert labels to one-hot encoding
y = to_categorical(y, 2)  # 2 is the number of classes (fire and non-fire)

# Step 2: Split the Dataset with Stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

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
try:
    loss, accuracy = model.evaluate(X_test, y_test)
    print("Test Accuracy:", accuracy)

    # Generate predictions
    y_pred = model.predict(X_test)

    # Convert predictions to labels (0 or 1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)

    # Create a confusion matrix
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    print("Confusion Matrix:")
    print(cm)

    # Print a classification report for more detailed metrics
    print("Classification Report:")
    print(classification_report(y_true_labels, y_pred_labels))
except Exception as e:
    print("Error during evaluation:", e)


# Step 6: Process Images and Ring Alarm for Fire Detection
for i in range(len(X_test)):
    # Load the next image
    current_image = X_test[i]

    # Perform prediction using your trained model
    predictions = model.predict(np.expand_dims(current_image, axis=0))

    # Check if the prediction indicates fire (you may need to adjust the threshold)
    if predictions[0, 1] > 0.50:
        print("Fire detected in image {}".format(i))

        # Ring the alarm using winsound (adjust the frequency and duration as needed)
        frequency = 2500  # Set frequency to 2500 Hertz
        duration = 1000  # Set duration to 2000 milliseconds (2 seconds)
        winsound.Beep(frequency, duration)
    else:
        print("No fire detected in image {}".format(i))

# End of the loop

# Display the final evaluation results
print("Final Test Accuracy:", accuracy)
