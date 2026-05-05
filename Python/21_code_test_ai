import cv2

clf = cv2.face.LBPHFaceRecognizer_create()
clf.read('face_train.yml')

faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cam = cv2.VideoCapture(0)

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        id, conf = clf.predict(gray[y:y+h, x:x+w])

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        cv2.putText(img, f'ID: {id}', (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.imshow('Face Recognition', img)

    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()
