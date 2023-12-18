import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import yaml


image = cv2.imread(r"C:\Users\safwa\Desktop\motherboard_image.JPEG")

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

min_contour_area = 100
_, thresholded = cv2.threshold(gray_image, 80, 255, cv2.THRESH_BINARY)
inverted_thresholded = cv2.bitwise_not(thresholded)
contours, _ = cv2.findContours(inverted_thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]
mask = np.zeros_like(image)
cv2.drawContours(mask, filtered_contours, -1, (255, 255, 255), thickness=cv2.FILLED)
result = cv2.bitwise_and(image, mask)

_, thresholded = cv2.threshold(gray_image, 90, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
mask = np.zeros_like(image)
cv2.drawContours(mask, filtered_contours, -1, (255, 255, 255), thickness=cv2.FILLED)
result1 = cv2.bitwise_and(result, mask)


plt.figure(figsize=(15, 10))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(result1, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.tight_layout()
plt.show()

#Step 2: YOLOv8 Training
# Load a YOLOv8 model
model = YOLO('yolov8m.pt')

# Train the model
results = model.train(data='C:/Users/safwa/PycharmProjects/pythonProject2/data.yaml', epochs=100, imgsz=1000, batch=4, name='AER850_Proj3_Model')


