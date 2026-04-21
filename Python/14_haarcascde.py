import cv2

# แก้ไข: ใช้ไฟล์ XML ที่มาพร้อมกับ OpenCV
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

# แก้ไข: ระบุพาธเต็มของรูปภาพ
img = cv2.imread(r'C:\Users\fkdjk\Downloads\Digital-Image-Processing\Python\girl.jpg')

# ตรวจสอบว่าพบไฟล์รูปภาพหรือไม่
if img is None:
    raise FileNotFoundError("ไม่พบไฟล์รูปภาพ กรุณาตรวจสอบพาธให้ถูกต้อง")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv2.imshow('face detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()