import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def detect_features(image, num_features=100000):
    """
    Detects features in an image using ORB.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(nfeatures=num_features)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors

# Paths to images
left_image_path = "left_disjoined.png"
right_image_path = "right_disjoined.png"

# Load images with error checking
left_image = cv2.imread(left_image_path, cv2.IMREAD_UNCHANGED)
right_image = cv2.imread(right_image_path, cv2.IMREAD_UNCHANGED)

if left_image is None or right_image is None:
    raise FileNotFoundError("Error: One or both images were not found. Check file paths!")

# Convert images to RGBA format to preserve transparency
if left_image.shape[-1] == 3:
    left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2BGRA)
if right_image.shape[-1] == 3:
    right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2BGRA)

# Detect features
keypoints_left, descriptors_left = detect_features(left_image)
keypoints_right, descriptors_right = detect_features(right_image)

# Draw keypoints with transparency handling
keypoint_img_left = cv2.drawKeypoints(left_image, keypoints_left, None, color=(0, 255, 0, 255), flags=0)
keypoint_img_right = cv2.drawKeypoints(right_image, keypoints_right, None, color=(0, 255, 0, 255), flags=0)

# Create transparency masks
mask_left = cv2.cvtColor(left_image[:, :, :3], cv2.COLOR_BGR2GRAY) > 0
mask_right = cv2.cvtColor(right_image[:, :, :3], cv2.COLOR_BGR2GRAY) > 0

keypoint_img_left[:, :, 3] = np.where(mask_left, 255, 0)  # Set transparency
keypoint_img_right[:, :, 3] = np.where(mask_right, 255, 0)

# Save images in the same directory with transparency
cv2.imwrite("keypoints_left.png", keypoint_img_left, [cv2.IMWRITE_PNG_COMPRESSION, 9])
cv2.imwrite("keypoints_right.png", keypoint_img_right, [cv2.IMWRITE_PNG_COMPRESSION, 9])

# Display images
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(cv2.cvtColor(keypoint_img_left, cv2.COLOR_BGRA2RGBA))
axes[0].set_title("Left Image Keypoints")
axes[0].axis("off")

axes[1].imshow(cv2.cvtColor(keypoint_img_right, cv2.COLOR_BGRA2RGBA))
axes[1].set_title("Right Image Keypoints")
axes[1].axis("off")

plt.tight_layout()
plt.savefig("feature_detection_output.png", transparent=True)
plt.show()

print("Feature detection complete! Keypoint images saved in the same directory with transparency.")
print("Feature visualization saved as: feature_detection_output.png")