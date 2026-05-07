import cv2
img = cv2.imread('Python/image.png')
newWidth = 600
newheight = 400
newImg = cv2.resize(img, (newWidth, newheight))
cv2.imshow('original image', img)
cv2.imshow('resized image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#img = cv2.imread(r'C:\Users\fkdjk\Downloads\Python Projcet\Python\image.png')