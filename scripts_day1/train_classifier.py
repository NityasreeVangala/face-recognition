import os
import numpy as np
import pickle
from sklearn.svm import SVC

embedding_dir = "embeddings_v3"
model_path = "models_v3/face_classifier_v3.pkl"

X = []
y = []

# 🔁 Load embeddings from .npy files
for filename in os.listdir(embedding_dir):
    if filename.endswith(".npy"):
        label = filename.split(".")[0]  # e.g., "nitya" from "nitya.npy"
        embed = np.load(os.path.join(embedding_dir, filename))  # shape: (N, 512)
        X.extend(embed)  # Add all N embeddings
        y.extend([label] * len(embed))  # Same label repeated N times

print(f"✅ Total embeddings loaded: {len(X)}")

# 🧠 Train SVM classifier
clf = SVC(kernel='linear', probability=True)
clf.fit(X, y)

# 💾 Save model
with open(model_path, "wb") as f:
    pickle.dump((clf, list(set(y))), f)

print(f"✅ Classifier trained and saved to {model_path}")


