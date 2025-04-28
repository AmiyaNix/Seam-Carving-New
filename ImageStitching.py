import cv2
import numpy as np
import os

def main():
    # Load the original image
    image_path = r"original_image.png"
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    bgr_img = cv2.imread(image_path)
    if bgr_img is None:
        raise FileNotFoundError(f"OpenCV Error: Could not load image '{image_path}'.")
    
    # Convert BGR to RGB for correct color representation in OpenCV
    original_image = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    
    # Save the final output image
    output_path = r"final_output.png"
    cv2.imwrite(output_path, cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
    print(f"Final output image saved at: {output_path}")
    
    # Display only the original image
    cv2.imshow("Final Output", cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)  # Wait until a key is pressed
    cv2.destroyAllWindows()  # Close the window

if __name__ == "__main__":
    main()
