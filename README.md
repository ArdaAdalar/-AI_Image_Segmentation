# AI_Image_Segmentation

# AI Image Segmentation and Analysis Project

This repository contains implementations for various image processing and segmentation algorithms as part of the **Comp448 HW1** assignment. The project focuses on using OpenCV and Python to preprocess, segment, and evaluate images, particularly for tasks like foreground extraction, cell location detection, and vessel segmentation.

---

## Overview

### Key Features:
1. **Foreground Mask Extraction**:
   - Extracts foreground regions from images using adaptive thresholding and morphological operations.
   - Handles noise removal and refines the mask for better segmentation.

2. **Cell Location Detection**:
   - Identifies cell centroids in images.
   - Uses contour analysis, morphological filtering, and centroid computation for precise localization.

3. **Vessel Segmentation**:
   - Enhances and segments blood vessels in fundus images.
   - Includes preprocessing, vessel enhancement, and postprocessing steps.

4. **Quantitative Metrics**:
   - Calculates precision, recall, and F-score for segmentation accuracy.
   - Compares results with gold standard masks for evaluation.

---

## File Descriptions

1. **p1.py**:
   - Implements the foreground mask extraction pipeline.
   - Uses adaptive thresholding and morphological operations for mask refinement.
   - Outputs metrics for precision, recall, and F-score.

2. **p2.py**:
   - Detects cell locations in segmented images.
   - Applies majority filtering and centroid calculations to locate cells.
   - Visualizes results and evaluates metrics against gold standard annotations.

3. **p4.py**:
   - Focuses on vessel segmentation in medical images.
   - Preprocesses images with CLAHE for contrast enhancement.
   - Applies morphological operations for vessel enhancement and thresholding.
   - Evaluates results using precision, recall, and F-score metrics.

4. **Reports**:
   - **Comp448 Q1 Report.pdf**: Details the foreground mask extraction process, including pseudocode, parameter discussions, and visual results.
   - **Comp448 Q2 Report.pdf**: Describes cell location detection techniques, highlighting contour-based approaches and metrics.
   - **Comp448 Q4 Report_.pdf**: Explains the vessel segmentation process, with a focus on parameter tuning and morphological operations.

5. **README.md**:
   - Contains instructions for running the code and adjusting paths.

---

## Requirements

- Python 3.x
- OpenCV
- NumPy
- Matplotlib

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/ArdaAdalar/AI_Search_Algorithms.git
   cd AI_Search_Algorithms
   
2. Install dependencies:
pip install -r requirements.txt
Adjust image paths in the scripts (p1.py, p2.py, p4.py) to match your local setup.

3. Run the scripts:

Foreground Mask Extraction:
  python p1.py
Cell Location Detection:
  python p2.py
Vessel Segmentation:
  python p4.py
## Results

Foreground Mask:

Binary masks showing the extracted foreground regions.
Metrics: Precision, Recall, F-Score.
Cell Locations:

Centroid positions visualized on images.
Metrics: Precision, Recall, F-Score.
Vessel Segmentation:

Enhanced vessel maps and segmented masks.
Metrics: Precision, Recall, F-Score.
Key Parameters
Foreground Mask Extraction:

Thresholding method: cv2.ADAPTIVE_THRESH_GAUSSIAN_C.
Block size: 11.
Kernel size for morphological operations: (5, 5).
Cell Location Detection:

Filter size: 3x3 kernel for majority filtering.
Vessel Segmentation:

CLAHE parameters: clipLimit=2.0, tileGridSize=(8,8).
Morphological kernel: cv2.MORPH_ELLIPSE with size (5, 5).
Thresholding method: Otsu's global thresholding.
