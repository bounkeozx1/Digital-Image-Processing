import cv2

# 1. โหลดภาพ
img = cv2.imread('python/gold.jpeg')

# 2. ปรับขนาดภาพ (สำคัญมาก: ต้องปรับก่อนวาด)
newWidth, newHeight = 600, 400
img = cv2.resize(img, (newWidth, newHeight))

# 3. วาดรูปร่าง (พิกัดต้องอยู่ในช่วง 0-600 และ 0-400)
# Arrow (ลูกศร)
cv2.arrowedLine(img, (50, 200), (150, 200), (0, 255, 0), 5)

# Line (เส้นตรง)
cv2.line(img, (10, 10), (210, 90), (155, 0, 0), 5)

# Rectangle (สี่เหลี่ยม - เปลี่ยนพิกัดให้พอดีกับภาพ 600x400)
cv2.rectangle(img, (200, 90), (500, 350), (0, 0, 255), 5)

# Circle (วงกลม)
cv2.circle(img, (350, 220), 120, (0, 255, 255), 5)

# Text (ข้อความ)
cv2.putText(img, 'gold', (315, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

# 4. แสดงผล
cv2.imshow('gold', img)
cv2.waitKey(0)
cv2.destroyAllWindows()