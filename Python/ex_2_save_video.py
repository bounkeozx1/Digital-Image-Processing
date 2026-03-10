import cv2

vdo = cv2.VideoCapture('Python/Video1.mp4')

count = 0

while True:
    check, frame = vdo.read()

    if not check:
        break

    cv2.imwrite(f'./image/video_frame_{count}.png', frame)

    count += 1

    cv2.imshow('video', frame)

    if cv2.waitKey(40) & count == 21:
        break

vdo.release()
cv2.destroyAllWindows()
