# User Manual: Seam Carving and Image Stitching Application

## Introduction
This application implements a pipeline for seam carving and image stitching. It processes images by detecting optimal seams, splitting images, identifying features, matching them, computing transformations, and finally stitching image parts together to create panoramas.

## System Requirements
- Python 3.6 or higher
- Required packages:
  - OpenCV (cv2)
  - NumPy
  - Matplotlib
  - PyTorch (for SSIM testing)
  - tkinter (for GUI components)

## Installation

1. Ensure Python is installed on your system
2. Install required packages:
```
pip install numpy opencv-python matplotlib torch tkinter
```

## Running the Application

### Method 1: Using the Main Script (Complete Pipeline)
The easiest way to run the complete pipeline is using the Main.py script:

1. Open a terminal/command prompt
2. Navigate to the project directory
3. Run:
```
python Main.py
```
4. The application will run all steps in sequence:
   - Seam carving
   - Feature detection
   - Feature matching
   - Homography estimation
   - Image warping
   - Image stitching

### Method 2: Running Individual Components

You can also run specific components independently:

#### Seam Carving
```
python NewSeamCarving.py
```
- When prompted, select an image file
- Outputs: original_image.png, energy_map.png, detected_seam.png, left_disjoined.png, right_disjoined.png

#### Feature Detection
```
python DetectFeatures.py
```
- Uses left_disjoined.png and right_disjoined.png as inputs
- Outputs: keypoints_left.png, keypoints_right.png, feature_detection_output.png

#### Feature Matching
```
python FeatureMatching.py
```
- Uses the previously generated keypoint images
- Outputs: matched_features_optimized.png

#### Homography Estimation
```
python EstimateHomography.py
```
- Computes and saves the transformation matrix
- Outputs: homography_matrix.txt

#### Image Warping
```
python ImageWarping.py
```
- Warps image parts using the homography matrix
- Outputs: warped_left_image.png, warped_right_image.png

#### Image Stitching
```
python ImageStitching.py
```
- Creates the final stitched image
- Outputs: final_output.png

#### SSIM Testing (Optional)
```
python ssimTest.py
```
- When prompted, select two images to compare
- Outputs a Structural Similarity Index value in the console

## Troubleshooting

### Common Issues

1. **Missing Files Error**
   - Ensure all required input files exist in the project directory
   - Run the scripts in the correct order

2. **Image Loading Errors**
   - Verify image files are not corrupted
   - Ensure images are in supported formats (PNG, JPG, JPEG, BMP)

3. **Not Enough Matches Error**
   - Try using an image with more distinctive features
   - Adjust feature detection parameters in DetectFeatures.py

4. **Memory Issues**
   - For large images, reduce the 'nfeatures' parameter in the ORB detector

## Output Files

The application generates several output files at different stages:
- Energy maps and seam visualizations
- Feature detection visualizations
- Matched features visualization
- Warped images
- Final stitched image

## Advanced Usage

- To compare the quality of different stitching results, use the ssimTest.py script
- To adjust the sensitivity of feature detection, modify the 'nfeatures' parameter in the ORB detector
- To change homography estimation parameters, modify the RANSAC threshold in EstimateHomography.py

## Technical Details

### Pipeline Overview
1. **Seam Carving**: Detects optimal seams in an image using energy maps and divides the image
2. **Feature Detection**: Uses ORB to find distinctive points in both image parts
3. **Feature Matching**: Matches corresponding features between the two image parts
4. **Homography Estimation**: Computes the transformation matrix using RANSAC
5. **Image Warping**: Applies the transformation to align the image parts
6. **Image Stitching**: Blends the transformed images to create a seamless panorama

### Key Algorithms
- **ORB Feature Detection**: Scale and rotation invariant feature detection
- **RANSAC**: Robust estimation of homography in presence of outliers
- **Perspective Warping**: Transformation of images using homography matrix
- **Alpha Blending**: Smooth transition between overlapping image regions

## Notes

- The application works best with images that have distinctive features
- For optimal results, use high-resolution images with minimal noise
- The process may take several minutes depending on image size and computer performance 