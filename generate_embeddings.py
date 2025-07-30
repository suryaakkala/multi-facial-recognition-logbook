import os
import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet

def preprocess_face(img, detector):
    results = detector.detect_faces(img)
    if results:
        x, y, w, h = results[0]['box']
        face = img[y:y+h, x:x+w]
        face = cv2.resize(face, (160, 160))
        return face
    return None

def generate_embeddings(image_dir='faces', output_dir='embeddings'):
    os.makedirs(output_dir, exist_ok=True)
    detector = MTCNN()
    embedder = FaceNet()
    embeddings = []
    names = []

    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(image_dir, filename)
            img = cv2.imread(path)
            face = preprocess_face(img, detector)
            if face is not None:
                embedding = embedder.embeddings([face])[0]
                embeddings.append(embedding)
                names.append(os.path.splitext(filename)[0])
                print(f"✅ Processed: {filename}")
            else:
                print(f"⚠️ Face not found in {filename}")

    np.savez_compressed(os.path.join(output_dir, 'face_embeddings.npz'),
                        embeddings=np.array(embeddings),
                        names=np.array(names))
    print(f"✅ Embeddings saved in '{output_dir}/face_embeddings.npz'")

if __name__ == '__main__':
    generate_embeddings()
