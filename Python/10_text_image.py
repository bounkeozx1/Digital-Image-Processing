import cv2
img = cv2.imread('python/tree.jpg')
cv2.putText(img, "Tree", (210, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
cv2.imshow('tree', img)
cv2.waitKey(0)
cv2.destroyAllWindows()