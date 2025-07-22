import cv2
import torch
import numpy as np
import torchvision.transforms as transforms

# Feature extractor using PyTorch ResNet
def extract_face_features(frame, model, device):
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Define transform pipeline
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Apply transform
    input_tensor = transform(frame_rgb).unsqueeze(0).to(device)

    # Inference
    model.eval()
    with torch.no_grad():
        features = model(input_tensor)

    return features.cpu().numpy().flatten()
