

import cv2
import numpy as np
from skimage import feature
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load your preprocessed fire and non-fire images, extract features (LBP, shape, etc.) as needed
# For example, you can use the code snippets provided earlier for LBP and shape analysis

# Example: Load LBP and shape features and corresponding labels
lbp_features = np.load('lbp_features.npy')  # Load LBP features (numpy array)
shape_features = np.load('shape_features.npy')  # Load shape features (numpy array)
labels = np.load('labels.npy')  # Load corresponding labels (1 for fire, 0 for non-fire)

# Concatenate LBP and shape features (or use any combination of features)
combined_features = np.concatenate((lbp_features, shape_features), axis=1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(combined_features, labels, test_size=0.2, random_state=42)

# Initialize and train the SVM classifier
svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)
svm_classifier.fit(X_train, y_train)

# Predict on the test set
predictions = svm_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# Now you can use the trained classifier to predict fire/non-fire regions in new images.
# Extract features from the regions and use svm_classifier.predict() to get predictions.

