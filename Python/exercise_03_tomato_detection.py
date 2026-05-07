import cv2
import numpy as np

img = cv2.imread('Python/tomato.jpeg')
lower = np.array([0, 20, 70])
upper = np.array([80, 100, 255])
mask = cv2.inRange(img, lower, upper)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for data in contours:
    area = cv2.contourArea(data)
    if area > 5000:
        x, y, w, h = cv2.boundingRect(data)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, 'Tomato', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

cv2.imshow('tomato', img)
cv2.waitKey(0)
cv2.destroyAllWindows()