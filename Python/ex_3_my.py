import cv2
img = cv2.imread("python/gold.jpeg")
newWidth = 600
newHeight = 400
img = cv2.resize(img, (newWidth, newHeight))
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
cv2.putText(img, "gold", (300, 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
cv2.line(img, (10, 10), (255, 50), BLUE, 5)
cv2.rectangle(img, (190, 50), (510, 380), RED, 5)
cv2.arrowedLine(img, (100, 220), (190, 220), GREEN, 5, tipLength=0.3)
cv2.circle(img, (350, 220), 150, YELLOW, 5)
cv2.imshow("Annotated Gold", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('tree', img)
