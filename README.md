# Facial Recognition Attendance System

A lightweight facial recognition attendance system that:
- Detects multiple faces in a single frame using MTCNN.
- Extracts embeddings using FaceNet (`keras-facenet`).
- Recognizes known faces and logs attendance with timestamp.
- Marks unrecognized faces distinctly.
- Works without a dedicated GPU (CPU-compatible).

---

## 🗂️ Folder Structure

```
project/
│
├── faces/               # Folder where you place face images (image name = user ID)
├── embeddings/          # Stores generated `.npz` file with embeddings
├── generate_embeddings.py
├── facial_recognition.py
├── requirements.txt
└── attendance.csv       # Auto-generated log file
```

---

## 🛠️ Installation

1. Clone or download this repo.
    ```bash
    git clone https://github.com/suryaakkala/multi-facial-recognition-logbook.git
    ```
2. Create a virtual environment:
    ```bash
    python -m venv face_env
    face_env\Scripts\activate  # Windows
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## ▶️ Usage

1. **Add face images** to the `faces/` folder.  
   File name should be the **user ID** (e.g., `22000xxxxx.jpg`).

2. **Generate Embeddings**:
    ```bash
    python generate_embeddings.py
    ```

3. **Start Recognition & Attendance**:
    ```bash
    python facial_recognition.py
    ```

Press `q` to quit the preview window.

---

## 📝 Notes
- Unrecognized faces are shown in **red** with label `"Unrecognised"`.
- Confidence score (cosine similarity) is displayed below each face.
- Attendance is marked once per session per user.

---

## 🧪 Tested On
- Windows 11
- Python 3.9
- No GPU required
