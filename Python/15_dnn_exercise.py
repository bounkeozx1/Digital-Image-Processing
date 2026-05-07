import cv2
import numpy as np

model_file = 'Python/DNN/opencv_face_detector_uint8.pb'
config_file = 'Python/DNN/opencv_face_detector.pbtxt'

net = cv2.dnn.readNetFromTensorflow(model_file, config_file)
#cap = cv2.VideoCapture('Python/Mark.mp4')
cap = cv2.VideoCapture('Python/Video.mp4')
while True:
    ret, img = cap.read()
    if not ret:
        break
    h, w = img.shape[:2]

    blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(300, 300))
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow('face detection', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
