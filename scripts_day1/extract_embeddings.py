import os
import torch
from PIL import Image
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
import numpy as np

# === Paths ===
FACES_DIR = "dataset_faces_v3"
EMBED_DIR = "embeddings_v3"
os.makedirs(EMBED_DIR, exist_ok=True)

# === Load Pretrained FaceNet Model ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# === Image Transform: Resize, Normalize ===
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # ImageNet-like normalization
])

# === Loop Through Each Person ===
for person_name in os.listdir(FACES_DIR):
    person_path = os.path.join(FACES_DIR, person_name)
    if not os.path.isdir(person_path):
        continue

    embeddings = []
    print(f"\n🔍 Extracting for: {person_name}")

    for filename in os.listdir(person_path):
        try:
            img_path = os.path.join(person_path, filename)
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                embedding = model(img_tensor)
            embeddings.append(embedding.squeeze().cpu().numpy())

        except Exception as e:
            print(f"❌ Failed {filename}: {e}")

    # Save all embeddings to .npy file
    embeddings = np.array(embeddings)
    np.save(os.path.join(EMBED_DIR, f"{person_name}.npy"), embeddings)
    print(f"✅ Saved {embeddings.shape[0]} embeddings for {person_name}")
