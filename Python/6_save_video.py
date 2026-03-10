import cv2
vdo = cv2.VideoCapture('Python/Video1.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
result = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
while True:
    check, frame = vdo.read()
    result.write(frame)
    cv2.imshow('video', frame)
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break
vdo.release()
result.release()
cv2.destroyAllWindows()