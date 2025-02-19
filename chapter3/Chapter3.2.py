import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "path_to_your_images"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

canny_edges = cv2.Canny(image, 100, 200)

_, otsu_binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

otsu_binary = cv2.bitwise_not(otsu_binary)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
opening = cv2.morphologyEx(otsu_binary, cv2.MORPH_OPEN, kernel)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opening, connectivity=8)
min_area = 500 
filtered = np.zeros_like(opening)
for i in range(1, num_labels):
    if stats[i, cv2.CC_STAT_AREA] >= min_area:
        filtered[labels == i] = 255

contours, _ = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
filled = np.zeros_like(filtered)
cv2.drawContours(filled, contours, -1, (255), thickness=cv2.FILLED)

closing = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel)

plt.figure(figsize=(12, 8)) 

plt.subplot(2, 3, 1)
plt.title("Original")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title("Edge detection")
plt.imshow(canny_edges, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title("Threshold segmentation")
plt.imshow(otsu_binary, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title("Opening")
plt.imshow(opening, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title("Filled")
plt.imshow(filled, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.title("Closing")
plt.imshow(closing, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
