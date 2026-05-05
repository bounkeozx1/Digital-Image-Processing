import numpy as np
from PIL import Image
import cv2, os

base_dir = os.path.dirname(os.path.abspath(__file__))
faces_dir = os.path.join(base_dir, 'faces')

if not os.path.exists(faces_dir):
    print("❌ ไม่พบโฟลเดอร์ faces:", faces_dir)
    exit()

image_paths = [os.path.join(faces_dir, f) for f in os.listdir(faces_dir)]

faces = []
ids = []

for image in image_paths:
    try:
        img = Image.open(image).convert('L')
        imageNP = np.array(img, 'uint8')

        id = int(os.path.split(image)[1].split(".")[0])

        faces.append(imageNP)
        ids.append(id)
    except Exception as e:
        print("ข้ามไฟล์:", image, "error:", e)

ids = np.array(ids)

clf = cv2.face.LBPHFaceRecognizer_create()
clf.train(faces, ids)
clf.write('face_train.yml')

print('✅ Train successful !!')
