import cv2
import numpy as np
import matplotlib.pyplot as plt

def estimate_homography(keypoints_left, keypoints_right, matches, reproj_thresh=4.0):
    """
    Estimates the homography matrix using RANSAC.

    Args:
        keypoints_left (list of cv2.KeyPoint): Keypoints from the left image.
        keypoints_right (list of cv2.KeyPoint): Keypoints from the right image.
        matches (list of cv2.DMatch): Matched keypoints.
        reproj_thresh (float): RANSAC re-projection threshold.

    Returns:
        homography_matrix (numpy array): 3x3 Homography matrix.
        mask (numpy array): Inlier mask used in RANSAC.
    """
    if len(matches) < 4:
        raise ValueError("Not enough matches to estimate homography. At least 4 required!")

    # Extract matched keypoints
    src_pts = np.float32([keypoints_left[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_right[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute homography using RANSAC
    homography_matrix, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, reproj_thresh)

    return homography_matrix, mask

# Load images
left_image = cv2.imread("left_disjoined.png", cv2.IMREAD_UNCHANGED)
right_image = cv2.imread("right_disjoined.png", cv2.IMREAD_UNCHANGED)

if left_image is None or right_image is None:
    raise FileNotFoundError("Error: One or both images were not found. Check file paths!")

# Convert images to RGB if they are in BGRA
if left_image.shape[-1] == 4:
    left_rgb = cv2.cvtColor(left_image, cv2.COLOR_BGRA2RGB)
else:
    left_rgb = left_image

if right_image.shape[-1] == 4:
    right_rgb = cv2.cvtColor(right_image, cv2.COLOR_BGRA2RGB)
else:
    right_rgb = right_image

# Detect ORB features
orb = cv2.ORB_create(nfeatures=100000)
keypoints_left, descriptors_left = orb.detectAndCompute(left_rgb, None)
keypoints_right, descriptors_right = orb.detectAndCompute(right_rgb, None)

# Ensure descriptors are not empty
if descriptors_left is None or descriptors_right is None:
    raise ValueError("Error: No descriptors found in one or both images!")

# Use Brute-Force Matcher (BFMatcher) with Hamming distance for ORB
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# Find matches
matches = bf.knnMatch(descriptors_left, descriptors_right, k=2)

# Apply Lowe's ratio test
good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

print(f"Good Matches Found: {len(good_matches)}")

# Ensure enough matches for homography
if len(good_matches) < 4:
    raise ValueError("⚠️ Not enough matches found to compute homography!")

# Estimate Homography
homography_matrix, mask = estimate_homography(keypoints_left, keypoints_right, good_matches)

# Save homography matrix
np.savetxt("homography_matrix.txt", homography_matrix)

print(f"Homography estimation complete! Matrix saved. \nHomography Matrix:\n{homography_matrix}")
