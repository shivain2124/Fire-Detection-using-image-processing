import cv2
import numpy as np

# Read the binary mask (preprocessed fire region)
binary_mask_path = 'binary_mask.jpg'
binary_mask = cv2.imread(binary_mask_path, cv2.IMREAD_GRAYSCALE)

# Define a kernel for morphological operations
kernel = np.ones((5, 5), np.uint8)  # Adjust kernel size based on the image resolution and noise level

# Apply dilation to merge nearby fire regions
dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)

# Apply erosion to reduce noise and separate merged regions
filtered_mask = cv2.erode(dilated_mask, kernel, iterations=1)

# Display the filtered mask
cv2.imshow('Filtered Mask', filtered_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Now, 'filtered_mask' contains the filtered binary mask with reduced noise and merged regions.
