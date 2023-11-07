import cv2

# Read the original color image
original_image_path = "C:\Python 3.10\data_image.jpg"
original_image = cv2.imread(original_image_path)

# Perform preprocessing (color-based segmentation, thresholding, etc.) to obtain the binary mask
# Example: Thresholding to create a binary mask (replace this with your actual preprocessing code)
grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
_, binary_mask = cv2.threshold(grayscale_image, 200, 255, cv2.THRESH_BINARY)

# Save the binary mask image
binary_mask_path = "binary_mask.jpg"
cv2.imwrite(binary_mask_path, binary_mask)

# Now, the binary mask image has been saved as 'binary_mask.jpg' in the current working directory.
