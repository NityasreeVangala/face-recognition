import torch
import pickle
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load models
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load trained classifier
with open("models_v3/face_classifier_v3.pkl", "rb") as f:
    classifier, class_names = pickle.load(f)

def predict_faces(image_path, threshold=0.80):
    img = Image.open(image_path).convert('RGB')
    boxes, _ = mtcnn.detect(img)
    faces = mtcnn(img)

    results = []
    if boxes is not None and faces is not None:
        for i, (box, face) in enumerate(zip(boxes, faces)):
            emb = resnet(face.unsqueeze(0).to(device)).detach().cpu()
            probs = classifier.predict_proba(emb)[0]
            confidence = max(probs)
            pred_idx = probs.argmax()
            name = class_names[pred_idx] if confidence >= threshold else "Unknown"
            results.append({
                "name": name,
                "confidence": round(confidence, 2),
                "box": [int(b) for b in box.tolist()]
            })
    return img, results
