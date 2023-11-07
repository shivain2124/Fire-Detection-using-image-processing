
import cv2
import numpy as np
from skimage import feature

# Read the input image (preprocessed fire region)
fire_region_path = "C:\Python 3.10\data_image.jpg"

fire_region = cv2.imread(fire_region_path, cv2.IMREAD_GRAYSCALE)

# Compute Local Binary Pattern (LBP) features
radius = 1  # LBP circular neighborhood radius
n_points = 8 * radius  # Number of points to sample on the circle
lbp_features = feature.local_binary_pattern(fire_region, n_points, radius, method='uniform')

# Calculate the histogram of LBP features
hist, _ = np.histogram(lbp_features.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))

# Normalize the histogram
hist = hist.astype("float")
hist /= (hist.sum() + 1e-7)

# Print or use the LBP histogram features for classification
print("LBP Features:")
print(hist)
