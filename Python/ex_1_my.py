import cv2
video = cv2.VideoCapture('Python/Video1.mp4')
while True:
    check, frame = video.read()
    if not check:
        print("End of video or cannot read the file.")
        break
    small_frame = cv2.resize(frame, (600, 400))
    cv2.imshow('Small Video', small_frame)
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyAllWindows()

#video = cv2.VideoCapture(r'C:\Users\fkdjk\Downloads\Python Projcet\Python\Video1.mp4')