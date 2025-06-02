import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from typing import Dict
import time
from torch.amp import autocast

# --- Model Definition ---
class CNNMobileNet(nn.Module):
    def __init__(self, num_attributes: int = 105, is_utkface: bool = False, dropout_rate: float = 0.5):
        super(CNNMobileNet, self).__init__()
        self.is_utkface = is_utkface
        self.backbone = models.mobilenet_v3_small(weights=None)  # Weights loaded later
        self.backbone.classifier = nn.Identity()
        for param in self.backbone.parameters():
            param.requires_grad = False

        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            dummy_features = self.backbone(dummy_input)
            if len(dummy_features.shape) == 4:
                dummy_features = nn.AdaptiveAvgPool2d(1)(dummy_features)
                dummy_features = dummy_features.flatten(1)
            elif len(dummy_features.shape) == 2:
                pass
            else:
                raise ValueError(f"Unexpected backbone output shape: {dummy_features.shape}")
            in_features = dummy_features.shape[1]

        self.cnn_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        if is_utkface:
            self.age_head = nn.Linear(512, 1)
            self.gender_head = nn.Linear(512, 2)
            self.race_head = nn.Linear(512, 5)
        else:
            self.head = nn.Linear(512, num_attributes)

    def forward(self, x: torch.Tensor) -> torch.Tensor or Dict[str, torch.Tensor]:
        features = self.backbone(x)
        if len(features.shape) == 4:
            features = nn.AdaptiveAvgPool2d(1)(features)
            features = features.flatten(1)
        elif len(features.shape) == 2:
            pass
        else:
            raise ValueError(f"Unexpected backbone output shape: {features.shape}")
        features = self.cnn_head(features)
        if self.is_utkface:
            return {
                'age': self.age_head(features).squeeze(-1),
                'gender': self.gender_head(features),
                'race': self.race_head(features)
            }
        output = torch.sigmoid(self.head(features))
        return output

# --- Preprocessing Transforms ---
def get_transforms():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# --- Inference Function ---
def predict(model, image: np.ndarray, transform, device: torch.device) -> Dict[str, any]:
    model.eval()
    # Convert BGR (OpenCV) to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Apply transforms
    image_tensor = transform(image_rgb).unsqueeze(0).to(device, non_blocking=True)
    
    with torch.no_grad(), autocast("cuda"):
        outputs = model(image_tensor)
    
    # Process outputs
    age = outputs['age'].item() * 116.0
    gender = torch.argmax(outputs['gender'], dim=1).item()
    race = torch.argmax(outputs['race'], dim=1).item()
    
    gender_label = 'Male' if gender == 0 else 'Female'
    race_labels = ['White', 'Black', 'Asian', 'Indian', 'Other']
    race_label = race_labels[race]
    
    return {
        'age': round(age, 1),
        'gender': gender_label,
        'race': race_label
    }

def main():
    # Configuration
    model_path = "models/best_CNN+MobileNet_lr0.001_bs64_dr0.2_wd0.0001_Adam_-3_utkface.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    update_interval = 1.0  # Update predictions every 1 second
    print(f"Using device: {device}")

    # Load model
    model = CNNMobileNet(is_utkface=True, dropout_rate=0.2)
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    model.to(device)
    model.eval()

    # Initialize webcam
    cap = cv2.VideoCapture(0)  # 0 for default webcam
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Get webcam properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam resolution: {width}x{height}")

    # Initialize transform
    transform = get_transforms()

    # Initialize timing
    last_update = time.time()
    predictions = {'age': 0.0, 'gender': 'Unknown', 'race': 'Unknown'}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break

            # Update predictions every second
            current_time = time.time()
            if current_time - last_update >= update_interval:
                try:
                    predictions = predict(model, frame, transform, device)
                    last_update = current_time
                except Exception as e:
                    print(f"Prediction error: {e}")

            # Overlay predictions on frame
            text = f"Age: {predictions['age']} | Gender: {predictions['gender']} | Race: {predictions['race']}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Display frame
            cv2.imshow('UTKFace Webcam Inference', frame)

            # Check for exit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        torch.cuda.empty_cache()
        print("Webcam released and resources cleaned up")

if __name__ == "__main__":
    main()