import cv2
import numpy as np

def warp_images(left_image, right_image, homography_matrix):
    """
    Warps the left and right images based on the homography matrix to align them.
    Converts the black background to transparent in the final panorama.

    Args:
        left_image (numpy array): Left image.
        right_image (numpy array): Right image.
        homography_matrix (numpy array): Homography matrix to align images.

    Returns:
        panorama (numpy array): Final stitched panorama image with transparency.
    """
    # Get image dimensions
    height_left, width_left = left_image.shape[:2]
    height_right, width_right = right_image.shape[:2]

    # Compute bounding box for both images
    corners_right = np.array([[0, 0], [width_right, 0], [0, height_right], [width_right, height_right]], dtype=np.float32)
    transformed_corners = cv2.perspectiveTransform(corners_right.reshape(-1, 1, 2), homography_matrix)

    x_min = min(transformed_corners[:, 0, 0].min(), 0)
    x_max = max(transformed_corners[:, 0, 0].max(), width_left + width_right)
    y_min = min(transformed_corners[:, 0, 1].min(), 0)
    y_max = max(transformed_corners[:, 0, 1].max(), max(height_left, height_right))

    panorama_width = int(x_max - x_min)
    panorama_height = int(y_max - y_min)

    # Translate images accordingly
    translation_matrix = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)

    warped_left_image = cv2.warpPerspective(left_image, translation_matrix @ homography_matrix, (panorama_width, panorama_height))
    warped_right_image = cv2.warpPerspective(right_image, translation_matrix, (panorama_width, panorama_height))

    # Convert to RGBA format (Add an Alpha channel)
    warped_left_image = cv2.cvtColor(warped_left_image, cv2.COLOR_BGR2BGRA)
    warped_right_image = cv2.cvtColor(warped_right_image, cv2.COLOR_BGR2BGRA)

    # Create transparency mask (set alpha = 0 for black areas)
    mask_left = (cv2.cvtColor(warped_left_image[:, :, :3], cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8) * 255
    mask_right = (cv2.cvtColor(warped_right_image[:, :, :3], cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8) * 255

    warped_left_image[:, :, 3] = mask_left  # Apply mask as Alpha channel
    warped_right_image[:, :, 3] = mask_right

    # Blend images using max operation
    panorama = np.maximum(warped_left_image, warped_right_image)

    return warped_left_image, warped_right_image, panorama

def main():
    # Load images
    left_image = cv2.imread("left_disjoined.png")
    right_image = cv2.imread("right_disjoined.png")

    if left_image is None or right_image is None:
        print("Error: Unable to load images. Ensure 'left_disjoined.png' and 'right_disjoined.png' exist.")
        return

    # Assuming a placeholder homography matrix
    homography_matrix = np.eye(3)

    # Warp and blend images
    warped_left_image, warped_right_image, panorama = warp_images(left_image, right_image, homography_matrix)

    # Save the final images with transparency
    cv2.imwrite("warped_left_image.png", warped_left_image)
    cv2.imwrite("warped_right_image.png", warped_right_image)
    cv2.imwrite("stitched_panorama.png", panorama)

    # Display output locally
    cv2.imshow("Warped Left Image", warped_left_image)
    cv2.imshow("Warped Right Image", warped_right_image)
    cv2.imshow("Stitched Panorama", panorama)

    cv2.waitKey(0)  # Wait until a key is pressed
    cv2.destroyAllWindows()  # Close image windows

if __name__ == "__main__":
    main()
