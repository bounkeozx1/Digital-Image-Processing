import cv2
img = cv2.imread('python/tree.jpg')
cv2.rectangle(img, (190, 170), (300, 290), (0, 255, 0), 5)
cv2.imshow('tree', img)
cv2.waitKey(0)
cv2.destroyAllWindows()