import cv2
import numpy as np
import matplotlib.pyplot as plt

def ObtainForegroundMask(image):
    # Preprocessing
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    

    # Adaptive thresholding to separate foreground and background
    mask = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Postprocessing
    # Apply morphological operations to remove small black areas and gaps
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours and remove small white areas in the background
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 1000:  
            cv2.drawContours(mask, [contour], -1, 0, cv2.FILLED)
    
    return mask



def CalculateMetrics(foreground_mask, gold_mask):
    true_positive = np.sum(np.logical_and(foreground_mask == 255, gold_mask == 1))
    false_positive = np.sum(np.logical_and(foreground_mask == 255, gold_mask == 0))
    false_negative = np.sum(np.logical_and(foreground_mask == 0, gold_mask == 1))
    
    if true_positive + false_positive == 0:
        precision = 0  # Handle division by zero
    else:
        precision = true_positive / (true_positive + false_positive)

    if true_positive + false_negative == 0: # Handle division by zero
        recall = 0  
    else:
        recall = true_positive / (true_positive + false_negative)

    if precision + recall == 0: # Handle division by zero
        f_score = 0  
    else:
        f_score = 2 * (precision * recall) / (precision + recall)
    
    return precision, recall, f_score

# Load and process each image
metrics_table = []

for i in range(1, 4):
    # Load image
    #to run the file on your local machine please change the img path
    image = cv2.imread(f"/Users/mehmetberkadalar/Downloads/data 2/im{i}.jpg")
    
    # Obtain foreground mask
    foreground_mask = ObtainForegroundMask(image)
    
    # Load gold standard mask
    #to run the file on your local machine please change the img path
    gold_mask = np.loadtxt(f"/Users/mehmetberkadalar/Downloads/data 2/im{i}_gold_mask.txt")
    
    # Calculate metrics
    precision, recall, f_score = CalculateMetrics(foreground_mask, gold_mask)
    metrics_table.append([precision, recall, f_score])
    
    # Plot image and estimated mask
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(foreground_mask, cmap='gray')
    plt.title('Estimated Mask')
    plt.axis('off')
    
    plt.suptitle(f'Image {i}')
    plt.show()

# Display metrics table
print("Image\tPrecision\tRecall\tF-score")
for i, metrics in enumerate(metrics_table, 1):
    precision, recall, f_score = metrics
    print(f"{i}\t{precision:.2f}\t\t{recall:.2f}\t{f_score:.2f}")







