import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

def select_image():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")])
    return file_path

def detect_seam(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    energy_map = np.abs(dx) + np.abs(dy)

    rows, cols = energy_map.shape
    M = energy_map.copy()
    backtrack = np.zeros_like(M, dtype=np.int32)

    for i in range(1, rows):
        for j in range(cols):
            min_energy = M[i-1, j]
            backtrack[i, j] = j

            if j > 0 and M[i-1, j-1] < min_energy:
                min_energy = M[i-1, j-1]
                backtrack[i, j] = j - 1

            if j < cols - 1 and M[i-1, j+1] < min_energy:
                min_energy = M[i-1, j+1]
                backtrack[i, j] = j + 1

            M[i, j] += min_energy

    seam = np.zeros(rows, dtype=np.int32)
    seam[rows - 1] = np.argmin(M[rows - 1])
    for i in range(rows - 2, -1, -1):
        seam[i] = backtrack[i + 1, seam[i + 1]]

    seam_img = img_rgb.copy()
    for i in range(rows):
        seam_img[i, seam[i]] = [255, 0, 0]

    return seam, energy_map, seam_img

def get_disjoined_sets(img_rgb, seam):
    rows, cols, channels = img_rgb.shape
    left_set = np.zeros((rows, cols, 4), dtype=np.uint8)
    right_set = np.zeros((rows, cols, 4), dtype=np.uint8)

    for i in range(rows):
        col = seam[i]
        left_set[i, :col, :3] = img_rgb[i, :col, :]
        left_set[i, :col, 3] = 255
        left_set[i, col:, 3] = 0

        right_set[i, col+1:, :3] = img_rgb[i, col+1:, :]
        right_set[i, col+1:, 3] = 255
        right_set[i, :col+1, 3] = 0

    return left_set, right_set

def main():
    image_path = select_image()
    if not image_path:
        print("No image selected. Exiting.")
        return

    bgr_img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    seam, energy_map, seam_img = detect_seam(img_rgb)
    left_set, right_set = get_disjoined_sets(img_rgb, seam)

    plt.imsave("original_image.png", img_rgb)
    plt.imsave("energy_map.png", energy_map, cmap="gray")
    plt.imsave("detected_seam.png", seam_img)
    plt.imsave("left_disjoined.png", left_set)
    plt.imsave("right_disjoined.png", right_set)
    print("Images saved!")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(energy_map, cmap="gray")
    axes[0, 1].set_title("Energy Map")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(seam_img)
    axes[0, 2].set_title("Detected Seam (Red)")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(left_set)
    axes[1, 0].set_title("Left Disjoined Set")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(right_set)
    axes[1, 1].set_title("Right Disjoined Set")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()




