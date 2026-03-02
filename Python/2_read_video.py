import cv2
video = cv2.VideoCapture(0)
while True:
    check, frame = video.read()
    cv2.imshow('Video', frame)
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyAllWindows()