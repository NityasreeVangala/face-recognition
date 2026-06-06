import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw, ImageFont
import pickle
import os

# --------- Configuration ---------
image_path = r"C:\Users\nitya\OneDrive\Desktop\face_id_system_v3\raw_photos\test4.jpg"
model_path = "models_v3/face_classifier_v3.pkl"
font_path = "arial.ttf"  # Use your system's font path if needed

# --------- Load Classifier ---------
with open(model_path, "rb") as f:
    classifier, class_names = pickle.load(f)

# --------- Load Test Image ---------
img = Image.open(image_path).convert('RGB')

# --------- Initialize Models ---------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=True, device=device)
embedder = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# --------- Detect Faces ---------
boxes, _ = mtcnn.detect(img)
draw = ImageDraw.Draw(img)

# Load font
try:
    font = ImageFont.truetype(font_path, 20)
except:
    font = ImageFont.load_default()

# --------- Recognize Each Face ---------
if boxes is not None:
    faces = mtcnn.extract(img, boxes, save_path=None)
    for idx, (box, face) in enumerate(zip(boxes, faces), 1):
        face = face.unsqueeze(0).to(device)
        embedding = embedder(face).detach().cpu()

        # Predict
        probs = classifier.predict_proba(embedding)[0]
        best_idx = torch.tensor(probs).argmax().item()
        name = class_names[best_idx]
        confidence = probs[best_idx]

        # Draw bounding box
        draw.rectangle(box.tolist(), outline='green', width=3)

        # Text positioning
        text = f'{name} ({confidence:.2f})'
        text_location = (box[0], box[1] - 25 if box[1] - 25 > 0 else box[1] + 5)
        draw.text(text_location, text, fill='green', font=font)

        # Console summary
        print(f"✅ Face {idx}: {name} (Confidence: {confidence:.2f})")
else:
    print("❌ No faces detected.")

# --------- Show Image ---------
img.show()




