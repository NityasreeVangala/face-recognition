import os
import cv2
from mtcnn.mtcnn import MTCNN

RAW_DIR = "dataset_raw_v3"
CROP_DIR = "dataset_faces_v3"
os.makedirs(CROP_DIR, exist_ok=True)

detector = MTCNN()

for person_name in os.listdir(RAW_DIR):
    person_path = os.path.join(RAW_DIR, person_name)
    if not os.path.isdir(person_path):
        continue

    person_crop_path = os.path.join(CROP_DIR, person_name)
    os.makedirs(person_crop_path, exist_ok=True)

    img_count = 0

    for img_file in os.listdir(person_path):
        img_path = os.path.join(person_path, img_file)

        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️ Skipped unreadable image: {img_path}")
                continue

            results = detector.detect_faces(img)

            if results:
                x, y, w, h = results[0]['box']
                x, y = max(0, x), max(0, y)
                w, h = max(1, w), max(1, h)  # Avoid zero dimensions

                # Skip if crop goes outside image
                if x + w > img.shape[1] or y + h > img.shape[0]:
                    print(f"❌ Skipped invalid crop: {img_path}")
                    continue

                face = img[y:y + h, x:x + w]

                if face.shape[0] == 0 or face.shape[1] == 0:
                    print(f"❌ Zero-sized face in: {img_path}")
                    continue

                resized_face = cv2.resize(face, (160, 160))
                save_path = os.path.join(person_crop_path, f"{img_count}.jpg")
                cv2.imwrite(save_path, resized_face)
                img_count += 1

        except Exception as e:
            print(f"❌ Error with {img_path}: {e}")

    print(f"✅ {person_name}: {img_count} faces saved to {person_crop_path}")

