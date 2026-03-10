import cv2
img = cv2.imread('python/tree.jpg')
cv2.circle(img, (250, 220), 50, (255, 0, 0), 5)
cv2.imshow('tree', img)
cv2.waitKey(0)
cv2.destroyAllWindows()