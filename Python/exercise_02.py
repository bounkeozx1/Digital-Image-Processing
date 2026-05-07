import cv2
video = cv2.VideoCapture('Python/Video1.mp4')
newwidth = 600
newheight = 400
count = 0
max_images = 20
while count < max_images:
    check, frame = video.read()
    if not check:
        break
    newImg = cv2.resize(frame, (newwidth, newheight))
    save_path = rf'./image/image_{count+1}.png'
    cv2.imwrite(save_path, newImg)
    cv2.imshow('resize', newImg)
    count += 1
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyAllWindows()

#video = cv2.VideoCapture(r'C:\Users\fkdjk\Downloads\Python Projcet\Python\Video1.mp4')

#    save_path = rf'C:\Users\fkdjk\Downloads\Python Projcet\image\image_{count+1}.png'
