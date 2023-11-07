
import cv2
import cv2

# Read the grayscale preprocessed fire region
fire_region_path = "C:\Python 3.10\data_image.jpg"

fire_region = cv2.imread(fire_region_path, cv2.IMREAD_GRAYSCALE)

# Find contours in the binary image
contours, _ = cv2.findContours(fire_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Read the original color image
original_image_path = "98C:\Python 3.10\data_image.jpg"
  # Provide the path to your original color image
original_image = cv2.imread(original_image_path)

# Loop over the contours and analyze shapes
for contour in contours:
    # Calculate area and perimeter of the contour
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    # Calculate circularity (4 * pi * area / perimeter^2)
    circularity = 0
    if perimeter > 0:
        circularity = (4 * 3.1416 * area) / (perimeter * perimeter)

    # Calculate aspect ratio (width / height)
    x, y, width, height = cv2.boundingRect(contour)
    aspect_ratio = float(width) / height

    # Filter contours based on circularity and aspect ratio thresholds
    circularity_threshold = 0.6  # Adjust based on application
    aspect_ratio_threshold = 1.0  # Adjust based on application

    if circularity > circularity_threshold and aspect_ratio > aspect_ratio_threshold:
        # Draw the contour on the original image
        cv2.drawContours(original_image, [contour], -1, (0, 255, 0), 2)  # Draw contours in green color

# Display the image with contours
cv2.imshow('Contours', original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
