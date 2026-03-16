import cv2
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

prototxt_path = os.path.join(MODEL_DIR, "deploy.prototxt")
model_path = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

def detect_face(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    net.setInput(blob)
    detections = net.forward()

    if detections.shape[2] == 0:
        return None

    max_conf = 0
    box = None

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > max_conf:
            max_conf = confidence
            box = detections[0, 0, i, 3:7]

    if max_conf < 0.6:
        return None


    box = box * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")

    startX = max(0, startX)
    startY = max(0, startY)
    endX = min(w, endX)
    endY = min(h, endY)

    face = frame[startY:endY, startX:endX]
    if face.size == 0:
        return None

    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (200, 200))

    return face

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)