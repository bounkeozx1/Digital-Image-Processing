import cv2
import numpy as np

# ตรวจสอบ path และนามสกุลไฟล์ให้ถูกต้อง
img = cv2.imread('Python/watermelon.jpeg')  # หรือ .jpg ตามไฟล์จริง

if img is None:
    print("ไม่พบไฟล์ภาพ ตรวจสอบ path และนามสกุลไฟล์")
else:
    img = cv2.resize(img, (1200, 800))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 100, 100])
    upper_green = np.array([80, 200, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 50000:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 3)

    cv2.imshow("result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
