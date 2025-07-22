import os
import sys
import json
import cv2
import torch
import joblib
import numpy as np
from tqdm import tqdm
from torchvision import models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.extract_features import extract_face_features

def load_dataset(dataset_dir, model, device, max_frames=5):
    features, labels = [], []

    for label_folder in os.listdir(dataset_dir):
        label_path = os.path.join(dataset_dir, label_folder)
        if not os.path.isdir(label_path):
            continue
        label = 1 if label_folder.lower() == "real" else 0

        for video_file in os.listdir(label_path):
            video_path = os.path.join(label_path, video_file)
            cap = cv2.VideoCapture(video_path)
            count = 0
            success, frame = cap.read()

            while success and count < max_frames:
                try:
                    feat = extract_face_features(frame, model, device)
                    features.append(feat)
                    labels.append(label)
                    count += 1
                except Exception as e:
                    print(f"[!] Error processing {video_file}: {e}")
                success, frame = cap.read()
            cap.release()

    return np.array(features), np.array(labels)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    base_model = models.resnet18(pretrained=True)
    feature_extractor = torch.nn.Sequential(*list(base_model.children())[:-1]).to(device)

    print("[INFO] Loading dataset...")
    X, y = load_dataset('dataset', feature_extractor, device)

    print("[INFO] Splitting dataset...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print("[INFO] Training classifier...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    print(f"[INFO] Validation Accuracy: {acc:.4f}")
    print(f"[INFO] Validation F1 Score: {f1:.4f}")

    os.makedirs('model', exist_ok=True)
    joblib.dump(clf, 'model/classifier.pkl')

    metrics = {
        "val_accuracy": round(acc * 100, 2),
        "val_f1": round(f1, 4)
    }
    with open('model/metrics.json', 'w') as f:
        json.dump(metrics, f)

    print("[INFO] Model and metrics saved.")
