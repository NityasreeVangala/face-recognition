import os
import torch
import pickle
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics import classification_report, confusion_matrix

# ========== CONFIG ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_dir = 'raw_photos'
classifier_path = 'models_v3/face_classifier_v3.pkl'
confidence_threshold = 0.80

# ========== LOAD MODELS ==========
mtcnn = MTCNN(keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
with open(classifier_path, 'rb') as f:
    classifier, class_names = pickle.load(f)

# ========== EVALUATE ==========
y_true, y_pred = [], []

for person_name in os.listdir(test_dir):
    person_folder = os.path.join(test_dir, person_name)
    if not os.path.isdir(person_folder):
        continue

    for img_file in os.listdir(person_folder):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(person_folder, img_file)
        img = Image.open(img_path).convert('RGB')
        box, prob = mtcnn.detect(img)

        if box is None:
            print(f"❌ No face in {img_path}")
            continue

        face = mtcnn(img)
        if face is None:
            continue

        emb = resnet(face.unsqueeze(0).to(device)).detach().cpu()
        probs = classifier.predict_proba(emb)[0]
        top_prob = max(probs)
        top_idx = probs.argmax()

        if top_prob >= confidence_threshold:
            pred_name = class_names[top_idx]
        else:
            pred_name = "Unknown"

        y_true.append(person_name)
        y_pred.append(pred_name)

        print(f"{img_file}: True = {person_name} | Pred = {pred_name} ({top_prob:.2f})")

# ========== REPORT ==========
print("\n🧾 Classification Report:")
print(classification_report(y_true, y_pred, labels=class_names + ["Unknown"], zero_division=0))

print("📊 Confusion Matrix:")
print(confusion_matrix(y_true, y_pred, labels=class_names + ["Unknown"]))
