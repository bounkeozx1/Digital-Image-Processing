import cv2
img = cv2.imread('Python/image.png')
newwifth = 600
nnewheigt = 400
newImg =cv2.resize(img, (newwifth, nnewheigt))
cv2.imwrite('Python/image_resized.png', newImg)
cv2.imshow('resize', newImg)
cv2.waitKey(0)
cv2.destroyAllWindows()

#img = cv2.imread(r'C:\Users\fkdjk\Downloads\Python Projcet\Python\image.png')

#cv2.imwrite(r'C:\Users\fkdjk\Downloads\Python Projcet\Python\image_resized.png', newImg)
