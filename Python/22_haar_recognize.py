import cv2
import os

# Build paths relative to the script's own location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

face_cascade = cv2.CascadeClassifier(os.path.join(BASE_DIR, 'haar cascade', 'haarcascade_frontalface_default.xml'))
cls = cv2.face.LBPHFaceRecognizer_create()
cls.read(os.path.join(BASE_DIR, '..', 'face_train.yml'))

label_names = {1: 'Person1', 2: 'Person2'}
cap = cv2.VideoCapture(os.path.join(BASE_DIR, 'video.mp4'))

while True:
    check, frame = cap.read()
    if not check or frame is None:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        current_face = gray[y:y+h, x:x+w]
        id_, conf = cls.predict(current_face)
        confident = round(100 - conf, 2)
        display_name = label_names.get(id_, f'ID: {id_}')
        text = f'{display_name} {confident}%'
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()