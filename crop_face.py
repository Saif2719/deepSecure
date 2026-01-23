from mtcnn import MTCNN
import cv2
import os

# create output folder
os.makedirs("output", exist_ok=True)

detector = MTCNN()

# read image
img = cv2.imread("test.jpg")

if img is None:
    print("Image not found")
    exit()

# detect faces
faces = detector.detect_faces(img)

if len(faces) == 0:
    print("No face detected")
    exit()

# take first face
x, y, w, h = faces[0]['box']

# fix negative values
x, y = max(0, x), max(0, y)

# crop face
face_crop = img[y:y+h, x:x+w]

# save cropped face
cv2.imwrite("output/face_crop.jpg", face_crop)

print("Face cropped and saved in output/face_crop.jpg")
