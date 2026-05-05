import cv2
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

model_file = os.path.join(script_dir, 'DNN', 'opencv_face_detector_uint8.pb')
config_file = os.path.join(script_dir, 'DNN', 'opencv_face_detector.pbtxt')
net = cv2.dnn.readNetFromTensorflow(model_file, config_file)

video = cv2.VideoCapture(os.path.join(script_dir, 'Mark.mp4'))
if not video.isOpened():
    raise FileNotFoundError("ไม่พบไฟล์ Mark.mp4 หรือเปิดไม่ได้")

faces_dir = os.path.join(script_dir, 'faces')
os.makedirs(faces_dir, exist_ok=True)

fps = int(video.get(cv2.CAP_PROP_FPS)) or 30  # fallback ถ้า fps = 0
frame_id = 0
count = 1
MAX_CAPTURES = 5

while True:
    ret, img = video.read()
    if not ret:
        break

    frame_id += 1

    # ประมวลผลทุก 1 วินาที (ทุก fps เฟรม)
    if frame_id % fps != 0:
        if cv2.waitKey(1) == 27:
            break
        continue

    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()

    face_found = False
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            # ✅ Clamp พิกัดไม่ให้เกินขอบภาพ
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue  # ✅ ข้ามถ้า crop ว่าง

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"Face #{count}  ({confidence*100:.0f}%)"
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # ✅ ชื่อไฟล์ถูกต้อง: img_1.jpg, img_2.jpg ...
            save_path = os.path.join(faces_dir, f'img_{count}.jpg')
            success = cv2.imwrite(save_path, crop)
            if success:
                print(f"✅ บันทึกรูปที่ {count}: {save_path}")
            else:
                print(f"⚠️ บันทึกไม่สำเร็จ: {save_path}")

            face_found = True
            count += 1
            break  # เอาแค่หน้าแรกที่มั่นใจสูงสุดต่อเฟรม

    # ✅ แสดงภาพนอก for loop
    if face_found:
        cv2.imshow('Original image', img)
        cv2.imshow('Crop image', crop)
        cv2.waitKey(500)

    if count > MAX_CAPTURES:
        print(f"\n🎉 แคปครบ {MAX_CAPTURES} รูปแล้ว")
        break

    if cv2.waitKey(1) == 27:
        break

video.release()
cv2.destroyAllWindows()