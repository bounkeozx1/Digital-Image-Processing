import cv2
import os

# Build path relative to the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(script_dir, 'girl.jpg')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
img = cv2.imread(img_path)

# Guard against missing file
if img is None:
    raise FileNotFoundError(f"Image not found at: {img_path}")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    crop_img = img[y:y + h, x:x + w]
    cv2.imwrite(os.path.join(script_dir, 'face.jpg'), crop_img)

cv2.imshow('Original image', img)
cv2.imshow('Crop image', crop_img)
cv2.waitKey(0)
cv2.destroyAllWindows()