import cv2
import numpy as np

vd = cv2.VideoCapture('python/Video1.mp4')

while True:
    check, frame = vd.read()

   

    lower_yellow = np.array([0, 200, 200]) 
    upper_yellow = np.array([45, 255, 255])

    mask = cv2.inRange(frame, lower_yellow, upper_yellow)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 1000:
            x, y, w, h = cv2.boundingRect(cnt)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

vd.release()
cv2.destroyAllWindows()