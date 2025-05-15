import os
import asyncio
import platform
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from typing import List, Tuple, Dict
# Note: pykan library is not available in Pyodide; included as placeholder for KAN implementation
# For actual KAN, use https://github.com/KindXiaoming/pykan outside Pyodide

# --- Dataset Classes ---

class PETADataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.num_attributes = 65  # PETA has 65 attributes

        # Load images and labels from 10 folders
        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            label_file = os.path.join(folder_path, "Label.txt")
            if not os.path.exists(label_file):
                continue

            # Read Label.txt: image_name label1 label2 ...
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < self.num_attributes + 1:
                        continue
                    image_name = parts[0]
                    image_path = os.path.join(folder_path, image_name)
                    if not os.path.exists(image_path):
                        continue
                    # Convert labels to binary (0/1)
                    label = [int(x) for x in parts[1:self.num_attributes + 1]]
                    self.image_paths.append(image_path)
                    self.labels.append(label)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

class UTKFaceDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Load images and parse labels from filenames
        for img_name in os.listdir(root_dir):
            if not img_name.endswith('.jpg'):
                continue
            parts = img_name.split('_')
            if len(parts) != 4:
                continue
            try:
                age = int(parts[0])  # 0-116
                gender = int(parts[1])  # 0=male, 1=female
                race = int(parts[2])  # 0=White, 1=Black, 2=Asian, 3=Indian, 4=Others
                image_path = os.path.join(root_dir, img_name)
                self.image_paths.append(image_path)
                self.labels.append((age, gender, race))
            except ValueError:
                continue

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_path = self.image_paths[idx]
        age, gender, race = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        labels = {
            'age': torch.tensor(age, dtype=torch.float32),
            'gender': torch.tensor(gender, dtype=torch.long),
            'race': torch.tensor(race, dtype=torch.long)
        }
        return image, labels

# --- Data Preprocessing ---

def get_transforms(dataset: str):
    if dataset == 'PETA':
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(227),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif dataset == 'UTKFace':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

# --- Model Definitions ---

class KANLayer(nn.Module):
    # Simplified KAN implementation (placeholder; use pykan for full version)
    def __init__(self, in_features: int, out_features: int):
        super(KANLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.SiLU()  # Sigmoid-weighted linear unit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        return self.activation(x)

class CNNKAN(nn.Module):
    def __init__(self, num_attributes: int = 65, is_utkface: bool = False):
        super(CNNKAN, self).__init__()
        self.is_utkface = is_utkface
        # CNN backbone (ResNet-50)
        self.backbone = models.resnet50(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Remove final FC layer

        # KAN head
        self.kan = nn.Sequential(
            KANLayer(in_features, 512),
            KANLayer(512, 256),
            KANLayer(256, num_attributes if not is_utkface else 512)
        )

        if is_utkface:
            # UTKFace heads: age (regression), gender (binary), race (5-class)
            self.age_head = nn.Linear(512, 1)
            self.gender_head = nn.Linear(512, 2)
            self.race_head = nn.Linear(512, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor or Dict[str, torch.Tensor]:
        features = self.backbone(x)
        features = self.kan(features)
        if self.is_utkface:
            return {
                'age': self.age_head(features).squeeze(-1),
                'gender': self.gender_head(features),
                'race': self.race_head(features)
            }
        return torch.sigmoid(features)  # PETA: multi-label

class CNNMobileNet(nn.Module):
    def __init__(self, num_attributes: int = 65, is_utkface: bool = False):
        super(CNNMobileNet, self).__init__()
        self.is_utkface = is_utkface
        # MobileNetV3 backbone
        self.backbone = models.mobilenet_v3_small(pretrained=True)
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier = nn.Identity()

        # Custom CNN head
        self.cnn_head = nn.Sequential(
            nn.Conv2d(in_features, 512, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        if is_utkface:
            self.age_head = nn.Linear(512, 1)
            self.gender_head = nn.Linear(512, 2)
            self.race_head = nn.Linear(512, 5)
        else:
            self.head = nn.Linear(512, num_attributes)

    def forward(self, x: torch.Tensor) -> torch.Tensor or Dict[str, torch.Tensor]:
        features = self.backbone(x)
        features = self.cnn_head(features)
        if self.is_utkface:
            return {
                'age': self.age_head(features).squeeze(-1),
                'gender': self.gender_head(features),
                'race': self.race_head(features)
            }
        return torch.sigmoid(self.head(features))

class MLCNN(nn.Module):
    def __init__(self, num_attributes: int = 65, is_utkface: bool = False):
        super(MLCNN, self).__init__()
        self.is_utkface = is_utkface
        # Simple CNN (placeholder; extend with 5-7 conv layers as in OpenPAR)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        # TODO: Compute in_features based on input size
        self.fc = nn.Linear(128 * 56 * 56, num_attributes if not is_utkface else 512)
        if is_utkface:
            self.age_head = nn.Linear(512, 1)
            self.gender_head = nn.Linear(512, 2)
            self.race_head = nn.Linear(512, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor or Dict[str, torch.Tensor]:
        features = self.backbone(x)
        features = self.fc(features)
        if self.is_utkface:
            return {
                'age': self.age_head(features).squeeze(-1),
                'gender': self.gender_head(features),
                'race': self.race_head(features)
            }
        return torch.sigmoid(features)

class CLIPPAR(nn.Module):
    def __init__(self, num_attributes: int = 65, is_utkface: bool = False):
        super(CLIPPAR, self).__init__()
        # Placeholder: Use Hugging Face CLIP (requires external library)
        # TODO: Load CLIP-ViT-B/32, add prompt module
        raise NotImplementedError("CLIP-PAR requires Hugging Face Transformers")

class MTANet(nn.Module):
    def __init__(self, num_attributes: int = 65, is_utkface: bool = False):
        super(MTANet, self).__init__()
        # Placeholder: Implement multi-step attention
        # TODO: Use ResNet-50 backbone, add attention module
        raise NotImplementedError("MTA-Net requires custom attention implementation")

# --- Training Functions ---

def train_peta(model, train_loader, val_loader, epochs: int = 10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()  # Binary cross-entropy for multi-label

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader)}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        print(f"Validation Loss: {val_loss/len(val_loader)}")

def train_utkface(model, train_loader, val_loader, epochs: int = 10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    age_criterion = nn.MSELoss()  # Regression for age
    gender_criterion = nn.CrossEntropyLoss()
    race_criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images = images.to(device)
            age, gender, race = labels['age'].to(device), labels['gender'].to(device), labels['race'].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            age_loss = age_criterion(outputs['age'], age)
            gender_loss = gender_criterion(outputs['gender'], gender)
            race_loss = race_criterion(outputs['race'], race)
            loss = age_loss + gender_loss + race_loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader)}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                age, gender, race = labels['age'].to(device), labels['gender'].to(device), labels['race'].to(device)
                outputs = model(images)
                age_loss = age_criterion(outputs['age'], age)
                gender_loss = gender_criterion(outputs['gender'], gender)
                race_loss = race_criterion(outputs['race'], race)
                loss = age_loss + gender_loss + race_loss
                val_loss += loss.item()
        print(f"Validation Loss: {val_loss/len(val_loader)}")

# --- Main Execution ---

async def main():
    # Dataset paths
    peta_root = "./Datasets/PETA"
    utkface_root = "./Datasets/UTK"

    # Load datasets
    peta_train_dataset = PETADataset(peta_root, transform=get_transforms('PETA'))
    utkface_train_dataset = UTKFaceDataset(utkface_root, transform=get_transforms('UTKFace'))

    # Split datasets (80% train, 20% val for simplicity)
    peta_train_size = int(0.8 * len(peta_train_dataset))
    peta_val_size = len(peta_train_dataset) - peta_train_size
    peta_train_dataset, peta_val_dataset = torch.utils.data.random_split(
        peta_train_dataset, [peta_train_size, peta_val_size])

    utkface_train_size = int(0.8 * len(utkface_train_dataset))
    utkface_val_size = len(utkface_train_dataset) - utkface_train_size
    utkface_train_dataset, utkface_val_dataset = torch.utils.data.random_split(
        utkface_train_dataset, [utkface_train_size, utkface_val_size])

    # DataLoaders
    peta_train_loader = DataLoader(peta_train_dataset, batch_size=32, shuffle=True)
    peta_val_loader = DataLoader(peta_val_dataset, batch_size=32)
    utkface_train_loader = DataLoader(utkface_train_dataset, batch_size=32, shuffle=True)
    utkface_val_loader = DataLoader(utkface_val_dataset, batch_size=32)

    # Initialize models
    models_to_test = [
        ('CNN+KAN', CNNKAN(num_attributes=65, is_utkface=False), CNNKAN(is_utkface=True)),
        ('CNN+MobileNet', CNNMobileNet(num_attributes=65, is_utkface=False), CNNMobileNet(is_utkface=True)),
        ('MLCNN', MLCNN(num_attributes=65, is_utkface=False), MLCNN(is_utkface=True)),
        ('CLIP-PAR', CLIPPAR(num_attributes=65, is_utkface=False), CLIPPAR(is_utkface=True)),
        ('MTA-Net', MTANet(num_attributes=65, is_utkface=False), MTANet(is_utkface=True))
    ]

    # Train models
    for name, peta_model, utkface_model in models_to_test:
        print(f"\nTraining {name} on PETA...")
        train_peta(peta_model, peta_train_loader, peta_val_loader, epochs=5)
        print(f"\nTraining {name} on UTKFace...")
        train_utkface(utkface_model, utkface_train_loader, utkface_val_loader, epochs=5)

# Run in Pyodide environment
if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    asyncio.run(main())