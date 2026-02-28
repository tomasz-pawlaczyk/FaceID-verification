import cv2
import argparse
from face_utils import detect_face, ensure_dir

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True)
parser.add_argument("--source", default=0)
args = parser.parse_args()

dataset_path = f"data/dataset/{args.name}"
ensure_dir(dataset_path)

cap = cv2.VideoCapture(int(args.source))
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    face = detect_face(frame)

    if face is not None:
        count += 1
        cv2.imwrite(f"{dataset_path}/{count}.jpg", face)
        cv2.putText(frame, str(count), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Collecting", frame)

    if cv2.waitKey(1) == 13 or count >= 150:
        break

cap.release()
cv2.destroyAllWindows()