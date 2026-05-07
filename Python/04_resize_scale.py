import cv2
img = cv2.imread('Python/image.png')
newScale = 0.5
newScale =cv2.resize(img, None, fx=newScale, fy=newScale)
cv2.imshow('Original', img)
cv2.imshow('Resized', img)
cv2.waitKey(0)
cv2.destroyAllWindows