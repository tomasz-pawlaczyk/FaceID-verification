# FaceID-verification

FaceID-verification is a real-time face identification system built with Python and OpenCV. It combines DNN-based face detection (ResNet SSD) with LBPH recognition to identify known users, reject unknown individuals, and handle no-face scenarios from a live webcam stream.  



## Project Overview

**FaceID-verification** is a real-time face identification and verification system built with Python and OpenCV.  
The application detects a face from a live webcam stream, extracts facial features, and identifies whether the detected person belongs to a set of known individuals or should be classified as an unknown user.

The system supports four decision states:

- **Known person detected** (e.g., Tomasz, Ania)
- **Unknown person detected**
- **No face detected**
- **Real-time confidence (distance) evaluation**

### How It Works

1. A live video stream is captured from the webcam.
2. A Deep Neural Network (DNN) face detector (ResNet SSD) identifies faces in the frame.
3. The detected face is cropped, normalized (grayscale, resized to 200×200), and passed to the recognizer.
4. An LBPH (Local Binary Patterns Histograms) model predicts the closest matching identity.
5. A threshold-based decision layer determines:
   - If the face matches a known identity
   - If the face should be classified as **Unknown**
   - If no face is present

### Data Collection

Data was collected using a dedicated collection script that:

- Captures webcam frames
- Detects faces using OpenCV DNN
- Crops and normalizes facial regions
- Stores images in a per-person directory structure

Each participant (including myself and several colleagues) provided 150–300 face samples under:

- Different lighting conditions
- Slight head rotations
- Natural facial expressions

This ensured realistic training data and better generalization during testing.

### Decision Logic

The system performs **1:N identification** with rejection capability:

- If no face is detected → `"No face detected"`
- If face is detected:
  - Predict `(label, confidence)`
  - If `confidence < threshold` → Known identity
  - If `confidence ≥ threshold` → `"Unknown person"`

This approach prevents forced classification and introduces a practical identity rejection mechanism.

---

## OpenCV importance

OpenCV plays a critical role in two stages:

### 1. Deep Learning Face Detection (DNN)

The system uses OpenCV's DNN module with a pre-trained ResNet SSD model:

```python
blob = cv2.dnn.blobFromImage(
    cv2.resize(frame, (300, 300)),
    1.0,
    (300, 300),
    (104.0, 177.0, 123.0)
)

net.setInput(blob)
detections = net.forward()
```

This provides:

- Robust frontal face detection
- Better stability than Haar cascades
- Improved handling of lighting and minor pose changes

---

### 2. Face Recognition with LBPH

The LBPH recognizer is used for identity classification:

```python
model = cv2.face.LBPHFaceRecognizer_create()
model.train(faces, labels)

label, confidence = model.predict(face)
```

Key characteristics:

- Histogram-based texture descriptor
- Efficient for small datasets
- Interpretable distance metric
- Suitable for local/offline authentication systems

---

## Project Structure

```
face_unlock/
│
├── data/
│   ├── dataset/
│   │   ├── Tomasz/
│   │   └── Ania/
│   └── model/
│
├── models/
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
│
├── src/
│   ├── collect.py
│   ├── train.py
│   ├── recognise.py
│   └── face_utils.py
│
└── README.md
```

### Responsibilities

- **collect.py** – captures and stores training data
- **train.py** – trains the LBPH model and saves it to disk
- **recognise.py** – performs real-time face identification
- **face_utils.py** – handles DNN-based face detection

The project follows a modular structure to ensure maintainability and scalability.

---

## How to Use

### 1. Install Dependencies

```bash
pip install opencv-python opencv-contrib-python numpy
```

### 2. Collect Data

Collect face samples for each person:

```bash
python src/collect.py --name Tomasz
python src/collect.py --name Ania
```

Images will be stored under:

```
data/dataset/<PersonName>/
```

---

### 3. Train the Model

```bash
python src/train.py
```

This generates:

```
data/model/model.xml
data/model/labels.pkl
```

---

### 4. Run Face Identification

```bash
python src/recognise.py
```

Possible outputs:

- **Detected: Tomasz**
- **Detected: Ania**
- **Unknown person**
- **No face detected**

---

## Professional Value

This project demonstrates:

- Real-time computer vision processing
- Deep learning model integration (OpenCV DNN)
- Face recognition pipeline design
- Threshold-based decision systems
- Modular architecture and model persistence
- Practical identity verification logic

It represents a complete end-to-end computer vision application suitable for authentication, access control, and embedded identity systems.
