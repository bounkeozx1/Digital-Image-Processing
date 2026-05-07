import cv2
img = cv2.imread("python/tree.jpg")
cv2.arrowedLine(img, (250, 90), (250, 170), (0, 0, 255), 5)
cv2.line(img, (450, 40), (450, 370), (155, 0, 0), 5)
cv2.imshow("Arrow", img)
cv2.waitKey(0)
cv2.destroyAllWindows()