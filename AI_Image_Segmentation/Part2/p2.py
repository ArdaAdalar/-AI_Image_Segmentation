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
        if area < 1000:  # Adjust threshold as needed
            cv2.drawContours(mask, [contour], -1, 0, cv2.FILLED)
    
    # Invert the mask to use properly in the FindCellLocations function
    inverted_mask = cv2.bitwise_not(mask)
    
    return inverted_mask

def FindCellLocations(image, foreground_mask, filter_size=3):
    # Find contours in the refined foreground mask
    contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Initialize centroids list
    centroids = []
    
    # Calculate centroids of contours
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroids.append((cX, cY))
       
    # Create a blank image to store the regional maxima map
    regional_maxima_map = np.zeros_like(foreground_mask, dtype=np.uint8)
    
    # Draw circles for each centroid
    for centroid in centroids:
        cv2.circle(regional_maxima_map, centroid, 5, 255, -1)
    
    # Apply majority filtering
    kernel = np.ones((filter_size, filter_size), np.uint8)
    filtered_mask = cv2.morphologyEx(regional_maxima_map, cv2.MORPH_CLOSE, kernel)
    
    return centroids, filtered_mask




def CalculateCellMetrics(centroids, gold_cells):
    print("Detected Centroids:", centroids)
    print("Gold Standard Cells:", gold_cells)
    
    # Convert list of centroids to set for efficient membership testing
    centroids_set = set(map(tuple, centroids))  # Convert each centroid to tuple
    
    # Initialize variables for metrics
    true_positives = 0
    
    # Loop through centroids and find the label of the gold standard cell for each centroid
    for centroid in centroids:
        # Find the label of the gold standard cell for the centroid
        cell_label = gold_cells[centroid[1], centroid[0]]  # Note the coordinate order (y, x)
        if cell_label != 0:  # Check if it's not background
            true_positives += 1
    
    print("True Positives:", true_positives)
    
    # Calculate precision, recall, and F-score
    if len(centroids) == 0:
        precision = 0
    else:
        precision = true_positives / len(centroids)
    
    total_gold_cells = np.count_nonzero(np.unique(gold_cells))  # Count unique non-zero labels in gold_cells
    if total_gold_cells == 0:
        recall = 0
    else:
        recall = true_positives / total_gold_cells
    
    if precision + recall == 0:
        f_score = 0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)
    
    return precision, recall, f_score



# Load and process each image
metrics_table_part2 = []

for i in range(1, 4):
    # Load image
    #to run the image on your local machine please change the img path
    image = cv2.imread(f"/Users/mehmetberkadalar/Downloads/data 2/im{i}.jpg") 
    
    # Obtain foreground mask using Part 1 function
    foreground_mask = ObtainForegroundMask(image)
    
    # Call Part 2 function to find cell locations
    centroids, regional_maxima_map = FindCellLocations(image, foreground_mask)
    
    # Load gold standard cell locations
    #to run the image on your local machine please change the img path
    gold_cells = np.loadtxt(f"/Users/mehmetberkadalar/Downloads/data 2/im{i}_gold_cells.txt").astype(np.int)

    
    # Calculate metrics
    precision, recall, f_score = CalculateCellMetrics(centroids, gold_cells)
    metrics_table_part2.append([precision, recall, f_score])
    
    # Visualize cell locations on the original image
    for centroid in centroids:
        cv2.circle(image, centroid, 5, (0, 255, 0), -1)
    
    # Display images
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f'Image {i} with Cell Locations')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(foreground_mask, cmap='gray')
    plt.title(f'Foreground Mask')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(regional_maxima_map, cmap='gray')
    plt.title(f'Regional Maxima Map')
    plt.axis('off')
    
    plt.show()

# Display metrics table for Part 2
print("Image\tPrecision\tRecall\tF-score")
for i, metrics in enumerate(metrics_table_part2, 1):
    precision, recall, f_score = metrics
    print(f"{i}\t{precision:.2f}\t\t{recall:.2f}\t{f_score:.2f}") 










