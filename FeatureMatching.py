import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def match_features(descriptors_left, descriptors_right):
    """
    Matches features between two sets of ORB descriptors using BFMatcher with Hamming distance.
    """
    if descriptors_left is None or descriptors_right is None:
        raise ValueError("Descriptors are missing! Ensure feature detection was successful.")

    # Use BFMatcher (Brute Force Matcher) with Hamming distance for ORB
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(descriptors_left, descriptors_right)

    # Sort matches by distance (lower distance = better match)
    matches = sorted(matches, key=lambda x: x.distance)

    return matches

# File paths (ensure these files exist in your working directory)
left_image_path = "left_disjoined.png"
right_image_path = "right_disjoined.png"

if not os.path.exists(left_image_path) or not os.path.exists(right_image_path):
    raise FileNotFoundError("One or both images were not found. Check file paths!")

# Load images with alpha channel
left_image = cv2.imread(left_image_path, cv2.IMREAD_UNCHANGED)
right_image = cv2.imread(right_image_path, cv2.IMREAD_UNCHANGED)

# Ensure images are in RGBA format for transparency handling
if left_image.shape[-1] == 3:
    left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2BGRA)
if right_image.shape[-1] == 3:
    right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2BGRA)

# Detect ORB features
orb = cv2.ORB_create(nfeatures=200000)
keypoints_left, descriptors_left = orb.detectAndCompute(left_image[:, :, :3], None)
keypoints_right, descriptors_right = orb.detectAndCompute(right_image[:, :, :3], None)

# Match features
matches = match_features(descriptors_left, descriptors_right)

# Draw matches with transparency
num_matches_to_show = min(1000, len(matches))
matched_img = cv2.drawMatches(left_image[:, :, :3], keypoints_left, right_image[:, :, :3], keypoints_right, matches[:num_matches_to_show], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Convert to RGBA and set background transparency
matched_img_rgba = cv2.cvtColor(matched_img, cv2.COLOR_BGR2BGRA)
mask = cv2.cvtColor(matched_img, cv2.COLOR_BGR2GRAY) > 0
matched_img_rgba[:, :, 3] = np.where(mask, 255, 0)

# Save the transparent matched feature image
match_output_path = "matched_features_optimized.png"
cv2.imwrite(match_output_path, matched_img_rgba, [cv2.IMWRITE_PNG_COMPRESSION, 9])

# Display matched features
plt.figure(figsize=(16, 8))
plt.imshow(cv2.cvtColor(matched_img_rgba, cv2.COLOR_BGRA2RGBA))
plt.title(f"Feature Matching ({len(matches)} Matches Found)")
plt.axis("off")
plt.show()

print(f"Feature matching complete! {len(matches)} matches found.")
print(f"Transparent matched feature image saved as: {match_output_path}")