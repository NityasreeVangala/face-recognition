import os
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN

# === Config ===
RAW_DIR = "dataset_raw_v3"
CROP_DIR = "dataset_faces_v3"
os.makedirs(CROP_DIR, exist_ok=True)

# === Load face detector ===
detector = MTCNN()

# === Process each person's folder ===
for person_name in os.listdir(RAW_DIR):
    person_path = os.path.join(RAW_DIR, person_name)
    if not os.path.isdir(person_path):
        continue

    print(f"\n🔍 Processing: {person_name}")
    save_folder = os.path.join(CROP_DIR, person_name)
    os.makedirs(save_folder, exist_ok=True)

    count = 0

    for file_name in os.listdir(person_path):
        file_path = os.path.join(person_path, file_name)

        try:
            img = cv2.imread(file_path)
            if img is None:
                print(f"⚠️ Skipped (Unreadable): {file_path}")
                continue

            results = detector.detect_faces(img)
            if not results:
                print(f"❌ No face found in {file_name}")
                continue

            # Take first detected face
            x, y, w, h = results[0]['box']
            x, y = max(0, x), max(0, y)
            w, h = max(1, w), max(1, h)

            face = img[y:y+h, x:x+w]

            # Skip if the crop is invalid
            if face.shape[0] < 20 or face.shape[1] < 20:
                print(f"⚠️ Skipped (Too Small): {file_name}")
                continue

            face = cv2.resize(face, (160, 160))
            out_path = os.path.join(save_folder, f"{count}.jpg")
            cv2.imwrite(out_path, face)
            count += 1

        except Exception as e:
            print(f"❌ Error on {file_name}: {e}")

    print(f"✅ Saved {count} faces in: {save_folder}")
