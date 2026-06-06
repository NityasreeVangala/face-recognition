import os
import cv2
import torch
import pickle
from PIL import Image, ImageDraw, ImageFont
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms

# ====== Configuration ======
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_folder = 'raw_photos'  # Folder with test images
output_folder = 'output_v3'  # Where to save results
classifier_path = 'models_v3/face_classifier_v3.pkl'
confidence_threshold = 0.80
min_gap = 0.10

# ====== Load models ======
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# ====== Load classifier ======
with open(classifier_path, 'rb') as f:
    classifier, class_names = pickle.load(f)

# ====== Ensure output folder exists ======
os.makedirs(output_folder, exist_ok=True)

# ====== Process each image ======
for img_name in os.listdir(test_folder):
    if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    image_path = os.path.join(test_folder, img_name)
    img = Image.open(image_path).convert('RGB')
    boxes, _ = mtcnn.detect(img)

    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    if boxes is not None:
        faces = mtcnn(img)
        for i, (box, face) in enumerate(zip(boxes, faces)):
            emb = resnet(face.unsqueeze(0).to(device)).detach().cpu()
            probs = classifier.predict_proba(emb)[0]

            sorted_probs, sorted_idx = torch.sort(torch.tensor(probs), descending=True)
            top1_prob = sorted_probs[0].item()
            top2_prob = sorted_probs[1].item()
            pred_idx = sorted_idx[0].item()

            if top1_prob >= confidence_threshold and (top1_prob - top2_prob) >= min_gap:
                name = class_names[pred_idx]
                color = "green"
            else:
                name = "Unknown"
                color = "red"

            # Draw box and label
            draw.rectangle(box.tolist(), outline=color, width=2)
            label = f"{name} ({top1_prob:.2f})"
            draw.text((box[0], box[1] - 10), label, font=font, fill="white")
            print(f"✅ Face {i+1}: {label}")

    else:
        print(f"🚫 No face detected in {img_name}")

    # ====== Save output image ======
    output_path = os.path.join(output_folder, img_name)
    img.save(output_path)
    print(f"💾 Saved: {output_path}\n")

