import cv2
import numpy as np
from mtcnn import MTCNN

detector = MTCNN()

def detect_facial_boundary_issue(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return False

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb)

    if len(faces) == 0:
        return False

    x, y, w, h = faces[0]["box"]
    face = rgb[y:y+h, x:x+w]

    if face.size == 0:
        return False

    gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    h, w = edges.shape
    border = edges[int(0.1*h):int(0.9*h), int(0.1*w):int(0.9*w)]

    boundary_edges = np.sum(edges) - np.sum(border)
    inner_edges = np.sum(border)

    if inner_edges == 0:
        return False

    ratio = boundary_edges / inner_edges

    return ratio > 1.5   # threshold

def detect_abnormal_eye_texture(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return False

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb)

    if len(faces) == 0:
        return False

    keypoints = faces[0]["keypoints"]
    left_eye = keypoints["left_eye"]
    right_eye = keypoints["right_eye"]

    def eye_sharpness(eye_center):
        x, y = eye_center
        eye = rgb[y-15:y+15, x-15:x+15]

        if eye.size == 0:
            return 0

        gray = cv2.cvtColor(eye, cv2.COLOR_RGB2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    left_var = eye_sharpness(left_eye)
    right_var = eye_sharpness(right_eye)

    avg_var = (left_var + right_var) / 2

    return avg_var > 150   # threshold

def detect_skin_tone_irregularity(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return False

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb)

    if len(faces) == 0:
        return False

    x, y, w, h = faces[0]["box"]
    face = rgb[y:y+h, x:x+w]

    if face.size == 0:
        return False

    hsv = cv2.cvtColor(face, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    skin_variance = np.var(s) + np.var(v)

    return skin_variance > 500

def detect_gan_artifacts(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    h, w = magnitude.shape
    center = magnitude[h//4:3*h//4, w//4:3*w//4]

    high_freq_energy = np.mean(magnitude) - np.mean(center)

    return high_freq_energy > 20
