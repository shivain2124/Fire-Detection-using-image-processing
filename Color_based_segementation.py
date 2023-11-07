import cv2
import numpy as np

# Read the input image
image_path = "C:\Python 3.10\data_image.jpg"
original_image = cv2.imread(image_path)

# Convert the image to HSV color space
hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

# Define lower and upper HSV thresholds for detecting fire color
lower_fire = np.array([0, 120, 70])  # Lower threshold for fire color in HSV
upper_fire = np.array([10, 255, 255])  # Upper threshold for fire color in HSV

# Threshold the image to get the binary mask of fire regions
fire_mask = cv2.inRange(hsv_image, lower_fire, upper_fire)

# Apply morphological operations to remove noise and enhance the mask
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, kernel)
fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)

# Bitwise AND the original image with the fire mask to extract fire regions
fire_regions = cv2.bitwise_and(original_image, original_image, mask=fire_mask)

# Display the original image, fire mask, and extracted fire regions
cv2.imshow('Original Image', original_image)
cv2.imshow('Fire Mask', fire_mask)
cv2.imshow('Detected Fire Regions', fire_regions)
cv2.waitKey(0)
cv2.destroyAllWindows()
