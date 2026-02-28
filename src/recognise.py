import cv2
import pickle
from face_utils import detect_face

model = cv2.face.LBPHFaceRecognizer_create()
model.read("data/model/model.xml")

with open("data/model/labels.pkl", "rb") as f:
    label_map = pickle.load(f)

cap = cv2.VideoCapture(0)

threshold = 45

while True:
    ret, frame = cap.read()
    if not ret:
        break

    face = detect_face(frame)

    if face is None:
        cv2.putText(frame, "No face detected", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    else:
        label, confidence = model.predict(face)

        if confidence < threshold:
            name = label_map[label]
            text = f"Detected: {name}"
            color = (0,255,0)
        else:
            text = "Person unknown"
            color = (0,0,255)

        cv2.putText(frame, f"Distance: {int(confidence)}", (50, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

        cv2.putText(frame, text, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Recognition", frame)

    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()