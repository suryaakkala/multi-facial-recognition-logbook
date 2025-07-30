import os
import cv2
import numpy as np
from datetime import datetime
from mtcnn import MTCNN
from keras_facenet import FaceNet

# Load embeddings
data = np.load('embeddings/face_embeddings.npz')
known_embeddings = data['embeddings']
known_names = data['names']

# Attendance log
attendance_file = 'attendance.csv'
marked_today = set()
if not os.path.exists(attendance_file):
    with open(attendance_file, 'w') as f:
        f.write('UserID,Time\n')

def mark_attendance(name):
    if name not in marked_today:
        with open(attendance_file, 'a') as f:
            f.write(f"{name},{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        marked_today.add(name)
        print(f"📋 Marked attendance for {name}")

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Setup
embedder = FaceNet()
detector = MTCNN()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 5)

print("🎥 Starting facial recognition. Press 'q' to exit...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = detector.detect_faces(frame)
    for res in results:
        x, y, w, h = res['box']
        face = frame[y:y+h, x:x+w]
        try:
            face = cv2.resize(face, (160, 160))
            embedding = embedder.embeddings([face])[0]

            # Compare with known faces
            similarities = [cosine_similarity(embedding, db_emb) for db_emb in known_embeddings]
            best_idx = int(np.argmax(similarities))
            best_score = similarities[best_idx]

            name = "Unrecognised"
            if best_score > 0.6:
                name = known_names[best_idx]
                mark_attendance(name)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0) if name != "Unrecognised" else (0, 0, 255), 2)
            cv2.putText(frame, f"{name} ({best_score:.2f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        except Exception as e:
            print("❌ Error processing face:", e)

    cv2.imshow("Face Recognition Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import os
import cv2
import numpy as np
from datetime import datetime
from mtcnn import MTCNN
from keras_facenet import FaceNet

# Load embeddings
data = np.load('embeddings/face_embeddings.npz')
known_embeddings = data['embeddings']
known_names = data['names']

# Attendance log
attendance_file = 'attendance.csv'
marked_today = set()
if not os.path.exists(attendance_file):
    with open(attendance_file, 'w') as f:
        f.write('UserID,Time\n')

def mark_attendance(name):
    if name not in marked_today:
        with open(attendance_file, 'a') as f:
            f.write(f"{name},{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        marked_today.add(name)
        print(f"📋 Marked attendance for {name}")

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Setup
embedder = FaceNet()
detector = MTCNN()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 5)

print("🎥 Starting facial recognition. Press 'q' to exit...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = detector.detect_faces(frame)
    for res in results:
        x, y, w, h = res['box']
        face = frame[y:y+h, x:x+w]
        try:
            face = cv2.resize(face, (160, 160))
            embedding = embedder.embeddings([face])[0]

            # Compare with known faces
            similarities = [cosine_similarity(embedding, db_emb) for db_emb in known_embeddings]
            best_idx = int(np.argmax(similarities))
            best_score = similarities[best_idx]

            name = "Unrecognised"
            if best_score > 0.6:
                name = known_names[best_idx]
                mark_attendance(name)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0) if name != "Unrecognised" else (0, 0, 255), 2)
            cv2.putText(frame, f"{name} ({best_score:.2f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        except Exception as e:
            print("❌ Error processing face:", e)

    cv2.imshow("Face Recognition Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
