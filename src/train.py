import cv2
import numpy as np
import os
import pickle
from face_utils import ensure_dir

dataset_root = "data/dataset"
model_dir = "data/model"
ensure_dir(model_dir)

faces = []
labels = []
label_map = {}
current_label = 0

for person in os.listdir(dataset_root):
    person_path = os.path.join(dataset_root, person)
    if not os.path.isdir(person_path):
        continue

    label_map[current_label] = person

    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        faces.append(img)
        labels.append(current_label)

    current_label += 1

model = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
model.train(faces, np.array(labels))

model.save("data/model/model.xml")

with open("data/model/labels.pkl", "wb") as f:
    pickle.dump(label_map, f)