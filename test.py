from mtcnn import MTCNN
import cv2

detector = MTCNN()

img = cv2.imread("test.jpg")

if img is None:
    print("Image not found")
else:
    faces = detector.detect_faces(img)
    print(faces)
