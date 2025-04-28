
# Seam Carving and Image Stitching Project

This project implements an automated pipeline for **seam carving**, **feature detection**, **feature matching**, **homography estimation**, **image warping**, and **image stitching**.  
The goal is to manipulate and stitch images intelligently based on detected features.

## Project Structure

The project consists of the following Python scripts, executed sequentially:

1. **NewSeamCarving.py**  
   Performs content-aware image resizing using seam carving.

2. **DetectFeatures.py**  
   Detects important features/keypoints in the images (e.g., using SIFT, ORB).

3. **FeatureMatching.py**  
   Matches features between different images to find corresponding points.

4. **EstimateHomography.py**  
   Estimates the homography matrix between matched images for alignment.

5. **ImageWarping.py**  
   Warps images based on the estimated homography to prepare for stitching.

6. **ImageStitching.py**  
   Stitches the warped images into a single panoramic output.

---

## How to Run

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/your-username/your-repository.git
   cd your-repository
   ```

2. **Prepare the Environment**  
   Install the required Python libraries (you can use `requirements.txt` if you create one).
   Example:
   ```bash
   pip install opencv-python numpy
   ```

3. **Execute the Main Runner**  
   Run the controller script to sequentially execute all steps:
   ```bash
   python run_pipeline.py
   ```

   (Your `run_pipeline.py` is the script you shared above.)

---

## Requirements

- Python 3.7+
- Libraries:
  - OpenCV (`opencv-python`)
  - NumPy
  - (Optionally) Matplotlib (for visualization)

Install dependencies using:
```bash
pip install -r requirements.txt
```

---

## Directory Structure

```plaintext
Seam Carving new/
│
├── NewSeamCarving.py
├── DetectFeatures.py
├── FeatureMatching.py
├── EstimateHomography.py
├── ImageWarping.py
├── ImageStitching.py
├── run_pipeline.py
└── README.md
```

---

## Notes

- Make sure all input images are placed in the appropriate folder (modify scripts if needed).
- Results will be saved as output images, usually in the working directory or a designated folder.
- Each script prints its progress for easier debugging.

---

## Future Improvements

- Add command-line arguments to specify image directories.
- Support different feature detectors and matchers.
- Add unit tests for each processing step.
- Build a GUI for easier usage.

---

## License

This project is licensed under the MIT License — feel free to use, modify, and distribute it!
