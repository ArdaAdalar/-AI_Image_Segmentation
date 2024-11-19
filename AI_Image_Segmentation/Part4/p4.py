import cv2
import numpy as np
import os

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return enhanced

def enhance_vessels(image):
    # Apply morphological operations for vessel enhancement
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return opened

def threshold_vessels(image):
    # Apply global thresholding
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def postprocess_vessels(mask):
    # Perform morphological operations to remove noise
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    return opening

def evaluate_vessel_segmentation(segmented_mask, gold_mask):
    # Calculate metrics
    true_positive = np.sum(np.logical_and(segmented_mask == 255, gold_mask == 1))
    false_positive = np.sum(np.logical_and(segmented_mask == 255, gold_mask == 0))
    false_negative = np.sum(np.logical_and(segmented_mask == 0, gold_mask == 1))
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f_score = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f_score

# List of image filenames
image_filenames = [
    'd4_h_gold.png',
    'd4_h.jpg',
    'd7_dr_gold.png',
    'd7_dr.jpg',
    'd11_g_gold.png',
    'd11_g.jpg'
]

# Create the results folder if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')

# Process each image
for filename in image_filenames:
    # Load fundus image and gold standard mask
    image_path = os.path.join('fundus', filename)
    gold_mask_path = os.path.join('fundus', filename.split('.')[0] + '_gold.png')
    image = cv2.imread(image_path)
    gold_mask = cv2.imread(gold_mask_path, cv2.IMREAD_GRAYSCALE)

    # Preprocess image
    preprocessed_image = preprocess_image(image)

    # Enhance vessels
    enhanced_vessels = enhance_vessels(preprocessed_image)

    # Threshold vessels
    thresholded_vessels = threshold_vessels(enhanced_vessels)

    # Postprocess vessel mask
    segmented_vessels = postprocess_vessels(thresholded_vessels)

    # Evaluate vessel segmentation
    precision, recall, f_score = evaluate_vessel_segmentation(segmented_vessels, gold_mask)

    print("Image:", filename)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F-score:", f_score)

    # Save the maps images to the results folder
    cv2.imwrite(os.path.join('results', f"{filename.split('.')[0]}_preprocessed.jpg"), preprocessed_image)
    cv2.imwrite(os.path.join('results', f"{filename.split('.')[0]}_enhanced_vessels.jpg"), enhanced_vessels)
    cv2.imwrite(os.path.join('results', f"{filename.split('.')[0]}_thresholded_vessels.jpg"), thresholded_vessels)
    cv2.imwrite(os.path.join('results', f"{filename.split('.')[0]}_segmented_vessels.jpg"), segmented_vessels)
