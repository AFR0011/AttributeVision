import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from typing import List, Tuple, Dict
from sklearn.metrics import f1_score
from itertools import product
from datetime import datetime

# --- Custom Collate Function ---
def custom_collate(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) < 4:  # Adjusted minimum batch size for smaller batches
        return None
    return torch.utils.data.default_collate(batch)

# --- Dataset Classes ---
class PETADataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.num_binary_attributes = 61
        self.colors = ['Black', 'Blue', 'Brown', 'Green', 'Grey', 'Orange', 'Pink', 'Purple', 'Red', 'White', 'Yellow']
        self.color_categories = ['upperBody', 'lowerBody', 'hair', 'footwear']
        self.num_color_attributes = len(self.color_categories) * len(self.colors)
        self.num_total_attributes = self.num_binary_attributes + self.num_color_attributes

        self.binary_attr_map = {
            'accessoryHeadphone': 0, 'personalLess15': 1, 'personalLess30': 2, 'personalLess45': 3,
            'personalLess60': 4, 'personalLarger60': 5, 'carryingBabyBuggy': 6, 'carryingBackpack': 7,
            'hairBald': 8, 'footwearBoots': 9, 'lowerBodyCapri': 10, 'carryingOther': 11,
            'carryingShoppingTro': 12, 'carryingUmbrella': 13, 'lowerBodyCasual': 14, 'upperBodyCasual': 15,
            'personalFemale': 16, 'carryingFolder': 17, 'lowerBodyFormal': 18, 'upperBodyFormal': 19,
            'accessoryHairBand': 20, 'accessoryHat': 21, 'lowerBodyHotPants': 22, 'upperBodyJacket': 23,
            'lowerBodyJeans': 24, 'accessoryKerchief': 25, 'footwearLeatherShoes': 26, 'upperBodyLogo': 27,
            'hairLong': 28, 'lowerBodyLongSkirt': 29, 'upperBodyLongSleeve': 30, 'lowerBodyPlaid': 31,
            'lowerBodyThinStripes': 32, 'carryingLuggageCase': 33, 'personalMale': 34,
            'carryingMessengerBag': 35, 'accessoryMuffler': 36, 'accessoryNothing': 37,
            'carryingNothing': 38, 'upperBodyNoSleeve': 39, 'upperBodyPlaid': 40, 'carryingPlasticBags': 41,
            'footwearSandals': 42, 'footwearShoes': 43, 'hairShort': 44, 'lowerBodyShorts': 45,
            'upperBodyShortSleeve': 46, 'lowerBodyShortSkirt': 47, 'footwearSneakers': 48,
            'footwearStocking': 49, 'upperBodyThinStripes': 50, 'upperBodySuit': 51,
            'carryingSuitcase': 52, 'lowerBodySuits': 53, 'accessorySunglasses': 54, 'upperBodySweater': 55,
            'upperBodyThickStripes': 56, 'lowerBodyTrousers': 57, 'upperBodyTshirt': 58,
            'upperBodyOther': 59, 'upperBodyVNeck': 60
        }
        self.color_attr_map = {}
        idx = 0
        for category in self.color_categories:
            for color in self.colors:
                self.color_attr_map[f"{category}{color}"] = idx
                idx += 1

        folder_path = os.path.join(root_dir, "TownCentre")
        print(f"Checking folder: {folder_path}")
        if not os.path.isdir(folder_path):
            print(f"Error: TownCentre folder not found at {folder_path}")
            return

        label_file = os.path.join(folder_path, "Label.txt")
        print(f"Checking labels file: {label_file}")
        if not os.path.exists(label_file):
            print(f"Error: Label.txt not found at {label_file}")
            return

        print(f"Files in TownCentre: {os.listdir(folder_path)}")

        valid_lines = 0
        skipped_lines = 0
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    print(f"Skipping line (too short): {line.strip()}")
                    skipped_lines += 1
                    continue
                image_name = parts[0]
                matching_images = [
                    os.path.join(folder_path, filename)
                    for filename in os.listdir(folder_path)
                    if filename.startswith(f"{image_name}_") and filename.lower().endswith(('.jpg', '.jpeg', '.png'))
                ]
                if not matching_images:
                    print(f"Skipping line (no images found for {image_name}_): {line.strip()}")
                    skipped_lines += 1
                    continue
                try:
                    binary_labels = [0] * self.num_binary_attributes
                    color_labels = [0] * self.num_color_attributes
                    for attr in parts[1:]:
                        if attr in self.binary_attr_map:
                            binary_labels[self.binary_attr_map[attr]] = 1
                        elif self.color_attr_map and attr in self.color_attr_map:
                            color_labels[self.color_attr_map[attr]] = 1
                        else:
                            print(f"Warning: Unknown attribute {attr} in {label_file}")
                    for image_path in matching_images:
                        self.image_paths.append(image_path)
                        self.labels.append(binary_labels + color_labels)
                    print(f"Found {len(matching_images)} images for ID {image_name}")
                    valid_lines += 1
                except Exception as e:
                    print(f"Skipping invalid line in {label_file}: {line.strip()} (Error: {e})")
                    skipped_lines += 1
        print(f"Processed {valid_lines} valid lines, skipped {skipped_lines} lines")
        print(f"Total images loaded: {len(self.image_paths)}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Skipping corrupted image: {img_path} (Error: {e})")
            return None
        if self.transform:
            image = self.transform(image)
        return image, labels

class UTKFaceDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for img_name in os.listdir(root_dir):
            if not img_name.endswith('.jpg'):
                continue
            parts = img_name.split('_')
            if len(parts) != 4:
                continue
            try:
                age = int(parts[0])
                gender = int(parts[1])
                race = int(parts[2])
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
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Skipping corrupted image: {img_path}")
            return None
        if self.transform:
            image = self.transform(image)
        labels = {
            'age': torch.tensor(age / 116.0, dtype=torch.float32),
            'gender': torch.tensor(gender, dtype=torch.long),
            'race': torch.tensor(race, dtype=torch.long)
        }
        return image, labels

# --- Data Preprocessing ---
def get_transforms(dataset: str, is_training: bool = True):
    if dataset == 'PETA':
        if is_training:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    elif dataset == 'UTKFace':
        if is_training:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

# --- Model Definitions ---
class CNNMobileNet(nn.Module):
    def __init__(self, num_attributes: int = 105, is_utkface: bool = False, dropout_rate: float = 0.5):
        super(CNNMobileNet, self).__init__()
        self.is_utkface = is_utkface
        self.backbone = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
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


# --- Custom Collate Functions ---
def custom_collate(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) < 4:  # For PETA
        return None
    return torch.utils.data.default_collate(batch)

def custom_collate_utkface(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return torch.utils.data.default_collate(batch)


# --- MLCNN Model ---
class MLCNN(nn.Module):
    def __init__(self, num_attributes: int = 105, is_utkface: bool = False, dropout_rate: float = 0.5, num_conv_layers: int = 3, hidden_units: int = 512):
        super(MLCNN, self).__init__()
        self.is_utkface = is_utkface
        self.num_conv_layers = num_conv_layers  # Set attribute
        self.hidden_units = hidden_units  # Set attribute
        conv_layers = []
        in_channels = 3
        out_channels = [64, 128, 256, 512][:num_conv_layers]
        for out_channel in out_channels:
            conv_layers.extend([
                nn.Conv2d(in_channels, out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ])
            in_channels = out_channel
        conv_layers.append(nn.Flatten())
        conv_layers.append(nn.Dropout(dropout_rate))
        self.backbone = nn.Sequential(*conv_layers)

        dummy_input = torch.zeros(1, 3, 224, 224)
        in_features = self.backbone(dummy_input).shape[1]
        self.fc = nn.Linear(in_features, hidden_units)
        self.fc_relu = nn.ReLU()
        self.fc_dropout = nn.Dropout(dropout_rate)

        if is_utkface:
            self.age_head = nn.Linear(hidden_units, 1)
            self.gender_head = nn.Linear(hidden_units, 2)
            self.race_head = nn.Linear(hidden_units, 5)
        else:
            self.head = nn.Linear(hidden_units, num_attributes)
    def forward(self, x: torch.Tensor) -> torch.Tensor or Dict[str, torch.Tensor]:
        features = self.backbone(x)
        features = self.fc(features)
        features = self.fc_relu(features)
        features = self.fc_dropout(features)
        if self.is_utkface:
            return {
                'age': self.age_head(features).squeeze(-1),
                'gender': self.gender_head(features),
                'race': self.race_head(features)
            }
        return torch.sigmoid(self.head(features))

# --- Training and Evaluation Functions ---
def train_peta(model, train_loader, val_loader, epochs: int = 15, model_name: str = 'model', lr: float = 0.001, optimizer_type: str = 'Adam', weight_decay: float = 0.0, batch_size: int = 8, dropout_rate: float = 0.5, results_file: str = 'hyperparameter_results.txt'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    model.to(device)
    if optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    criterion = nn.BCELoss().to(device)
    torch.cuda.empty_cache()  # Clear GPU memory
    best_val_loss = float('inf')
    patience = 5
    counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for i, batch in enumerate(train_loader):
            if batch is None:
                print(f"Batch {i} is None, skipping")
                continue
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else float('inf')
        print(f"Epoch {epoch+1}, Train Loss: {train_loss}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss}")

        if (epoch + 1) % 5 == 0:
            metrics = evaluate_peta(model, val_loader, model_name, num_binary=61, num_colors=44)
            with open(results_file, 'a') as f:
                f.write(f"\nRun at {datetime.now()} (Epoch {epoch+1})\n")
                f.write(f"Dataset: PETA\n")
                f.write(f"Model: {model_name}\n")
                f.write(f"Hyperparameters:\n")
                f.write(f"  Learning Rate: {lr}\n")
                f.write(f"  Batch Size: {batch_size}\n")
                f.write(f"  Dropout Rate: {dropout_rate}\n")
                f.write(f"  Weight Decay: {weight_decay}\n")
                f.write(f"  Epoch: {epoch+1}\n")
                f.write(f"  Optimizer: {optimizer_type}\n")
                if model_name.startswith('MLCNN'):
                    f.write(f"  Conv Layers: {model.num_conv_layers}\n")
                    f.write(f"  Hidden Units: {model.hidden_units}\n")
                f.write(f"Validation Loss: {val_loss:.6f}\n")
                f.write(f"Evaluation Metrics:\n")
                for metric, value in metrics.items():
                    f.write(f"  {metric}: {value:.6f}\n")
                f.write("-" * 50 + "\n")
            print(f"Epoch {epoch+1} Metrics: {metrics}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), f"best_{model_name}_peta.pt")
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping")
                break
    return best_val_loss

def train_utkface(model, train_loader, val_loader, epochs: int = 15, model_name: str = 'model', lr: float = 0.001, optimizer_type: str = 'Adam', weight_decay: float = 0.0, batch_size: int = 8, dropout_rate: float = 0.5, results_file: str = 'hyperparameter_results.txt'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    model.to(device)
    if optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    age_criterion = nn.MSELoss().to(device)
    gender_criterion = nn.CrossEntropyLoss().to(device)
    race_criterion = nn.CrossEntropyLoss().to(device)
    torch.cuda.empty_cache()
    best_val_loss = float('inf')
    patience = 5
    counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            if batch is None:
                continue
            images, labels = batch
            images = images.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}
            optimizer.zero_grad()
            outputs = model(images)
            age_loss = age_criterion(outputs['age'], labels['age'])
            gender_loss = gender_criterion(outputs['gender'], labels['gender'])
            race_loss = race_criterion(outputs['race'], labels['race'])
            loss = 0.1 * age_loss + gender_loss + race_loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else float('inf')
        print(f"Epoch {epoch+1}, Train Loss: {train_loss}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                images, labels = batch
                images = images.to(device)
                labels = {k: v.to(device) for k, v in labels.items()}
                outputs = model(images)
                age_loss = age_criterion(outputs['age'], labels['age'])
                gender_loss = gender_criterion(outputs['gender'], labels['gender'])
                race_loss = race_criterion(outputs['race'], labels['race'])
                loss = 0.1 * age_loss + gender_loss + race_loss
                val_loss += loss.item()
        val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss}")

        if (epoch + 1) % 5 == 0:
            metrics = evaluate_utkface(model, val_loader, model_name)
            with open(results_file, 'a') as f:
                f.write(f"\nRun at {datetime.now()} (Epoch {epoch+1})\n")
                f.write(f"Dataset: UTKFace\n")
                f.write(f"Model: {model_name}\n")
                f.write(f"Hyperparameters:\n")
                f.write(f"  Learning Rate: {lr}\n")
                f.write(f"  Batch Size: {batch_size}\n")
                f.write(f"  Dropout Rate: {dropout_rate}\n")
                f.write(f"  Weight Decay: {weight_decay}\n")
                f.write(f"  Epoch: {epoch+1}\n")
                f.write(f"  Optimizer: {optimizer_type}\n")
                if model_name.startswith('MLCNN'):
                    f.write(f"  Conv Layers: {model.num_conv_layers}\n")
                    f.write(f"  Hidden Units: {model.hidden_units}\n")
                f.write(f"Validation Loss: {val_loss:.6f}\n")
                f.write(f"Evaluation Metrics:\n")
                for metric, value in metrics.items():
                    f.write(f"  {metric}: {value:.6f}\n")
                f.write("-" * 50 + "\n")
            print(f"Epoch {epoch+1} Metrics: {metrics}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), f"best_{model_name}_utkface.pt")
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping")
                break
    return best_val_loss

def evaluate_peta(model, loader, model_name: str, num_binary: int = 61, num_colors: int = 44):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            images, lbls = batch
            images, lbls = images.to(device), lbls.to(device)
            outputs = model(images)
            preds.append((outputs > 0.5).float().cpu().numpy())
            labels.append(lbls.cpu().numpy())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    binary_preds = preds[:, :num_binary]
    binary_labels = labels[:, :num_binary]
    color_preds = preds[:, num_binary:]
    color_labels = labels[:, num_binary:]
    metrics = {
        'f1_binary': f1_score(binary_labels, binary_preds, average='micro')
    }
    colors = ['Black', 'Blue', 'Brown', 'Green', 'Grey', 'Orange', 'Pink', 'Purple', 'Red', 'White', 'Yellow']
    color_categories = ['upperBody', 'lowerBody', 'hair', 'footwear']
    for i, category in enumerate(color_categories):
        start_idx = i * len(colors)
        end_idx = (i + 1) * len(colors)
        cat_preds = color_preds[:, start_idx:end_idx]
        cat_labels = color_labels[:, start_idx:end_idx]
        metrics[f'{category}_acc'] = f1_score(cat_labels, cat_preds, average='micro')
    return metrics
def evaluate_utkface(model, loader, model_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    age_errors, gender_preds, gender_labels, race_preds, race_labels = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            images, labels = batch
            images = images.to(device)
            outputs = model(images)
            age_errors.append(torch.abs(outputs['age'] - labels['age'].to(device)).cpu().numpy())
            gender_preds.append(torch.argmax(outputs['gender'], dim=1).cpu().numpy())
            gender_labels.append(labels['gender'].cpu().numpy())
            race_preds.append(torch.argmax(outputs['race'], dim=1).cpu().numpy())
            race_labels.append(labels['race'].cpu().numpy())
    age_mae = np.concatenate(age_errors).mean() * 116.0
    gender_acc = (np.concatenate(gender_preds) == np.concatenate(gender_labels)).mean()
    race_acc = (np.concatenate(race_preds) == np.concatenate(race_labels)).mean()
    metrics = {
        'age_mae': age_mae,
        'gender_acc': gender_acc,
        'race_acc': race_acc
    }
    return metrics

# --- Main Execution ---
def main():
    peta_root = "./Datasets/PETA"
    utkface_root = "./Datasets/UTK"

    print(f"Current working directory: {os.getcwd()}")
    peta_train_dataset = PETADataset(peta_root, transform=get_transforms('PETA', is_training=True))
    peta_val_dataset = PETADataset(peta_root, transform=get_transforms('PETA', is_training=False))
    print(f"PETA dataset size: {len(peta_train_dataset)}")

    if len(peta_train_dataset) == 0:
        raise ValueError("PETA dataset is empty. Check TownCentre folder and Label.txt.")

    utkface_train_dataset = UTKFaceDataset(utkface_root, transform=get_transforms('UTKFace', is_training=True))
    utkface_val_dataset = UTKFaceDataset(utkface_root, transform=get_transforms('UTKFace', is_training=False))
    print(f"UTKFace dataset size: {len(utkface_train_dataset)}")

    if len(utkface_train_dataset) == 0:
        raise ValueError("UTKFace dataset is empty. Check UTK folder.")

    peta_train_size = int(0.8 * len(peta_train_dataset))
    peta_val_size = len(peta_train_dataset) - peta_train_size
    print(f"PETA train size: {peta_train_size}, PETA val size: {peta_val_size}")
    if peta_val_size == 0:
        raise ValueError("PETA validation set is empty. Increase dataset size or adjust split ratio.")
    peta_train_dataset, peta_val_dataset = torch.utils.data.random_split(
        peta_train_dataset, [peta_train_size, peta_val_size])

    utkface_train_size = int(0.8 * len(utkface_train_dataset))
    utkface_val_size = len(utkface_train_dataset) - utkface_train_size
    print(f"UTKFace train size: {utkface_train_size}, UTKFace val size: {utkface_val_size}")
    if utkface_val_size == 0:
        raise ValueError("UTKFace validation set is empty. Increase dataset size or adjust split ratio.")
    utkface_train_dataset, utkface_val_dataset = torch.utils.data.random_split(
        utkface_train_dataset, [utkface_train_size, utkface_val_size])

    learning_rates = [0.005, 0.001]
    batch_sizes = [16, 32]
    dropout_rates = [0.2, 0.3]
    weight_decays = [0.0001]
    optimizers = ['Adam']
    mlcnn_configs = [
        {'name': 'MLCNN_base', 'num_conv_layers': 3, 'hidden_units': 512},
        {'name': 'MLCNN_large', 'num_conv_layers': 4, 'hidden_units': 1024}
    ]
    models_to_test = [
        ('CNN+MobileNet', CNNMobileNet, {'num_conv_layers': None, 'hidden_units': None}),
    ] + [(config['name'], MLCNN, config) for config in mlcnn_configs]

    results_file = 'hyperparameter_results.txt'
    with open(results_file, 'a') as f:
        f.write(f"\n=== Hyperparameter Search Started at {datetime.now()} ===\n")
        f.write(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}\n")
    for model_name, model_class, config in models_to_test:
        print(f"\nHyperparameter search for {model_name} on UTKFace...")
        for lr, batch_size, dropout_rate, weight_decay, optimizer_type in product(
            learning_rates, batch_sizes, dropout_rates, weight_decays, optimizers
        ):
            print(f"\nTesting: lr={lr}, batch_size={batch_size}, dropout_rate={dropout_rate}, weight_decay={weight_decay}, optimizer={optimizer_type}")
            
            utkface_train_loader = DataLoader(utkface_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_utkface, drop_last=True, pin_memory=True, num_workers=4)
            utkface_val_loader = DataLoader(utkface_val_dataset, batch_size=batch_size, collate_fn=custom_collate_utkface, drop_last=True, pin_memory=True, num_workers=4)

            if model_name.startswith('MLCNN'):
                utkface_model = model_class(
                    is_utkface=True,
                    dropout_rate=dropout_rate,
                    num_conv_layers=config['num_conv_layers'],
                    hidden_units=config['hidden_units']
                )
            else:
                utkface_model = model_class(is_utkface=True, dropout_rate=dropout_rate)
            
            model_id = f"{model_name}_lr{lr}_bs{batch_size}_dr{dropout_rate}_wd{weight_decay}_{optimizer_type}"
            best_val_loss = train_utkface(
                utkface_model,
                utkface_train_loader,
                utkface_val_loader,
                epochs=15,
                model_name=model_id,
                lr=lr,
                optimizer_type=optimizer_type,
                weight_decay=weight_decay,
                batch_size=batch_size,
                dropout_rate=dropout_rate,
                results_file=results_file
            )
            print(f"Results saved for {model_id}")
    else:
        print("Unknown optimizer type")

    for model_name, model_class, config in models_to_test:
        print(f"\nHyperparameter search for {model_name} on PETA...")
        for lr, batch_size, dropout_rate, weight_decay, optimizer_type in product(
            learning_rates, batch_sizes, dropout_rates, weight_decays, optimizers
        ):
            print(f"\nTesting: lr={lr}, batch_size={batch_size}, dropout_rate={dropout_rate}, weight_decay={weight_decay}, optimizer={optimizer_type}")
            
            peta_train_loader = DataLoader(peta_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate, drop_last=True, pin_memory=True, num_workers=4)
            peta_val_loader = DataLoader(peta_val_dataset, batch_size=batch_size, collate_fn=custom_collate, drop_last=True, pin_memory=True, num_workers=4)

            if model_name.startswith('MLCNN'):
                peta_model = model_class(
                    num_attributes=105,
                    is_utkface=False,
                    dropout_rate=dropout_rate,
                    num_conv_layers=config['num_conv_layers'],
                    hidden_units=config['hidden_units']
                )
            else:
                peta_model = model_class(num_attributes=105, is_utkface=False, dropout_rate=dropout_rate)
            
            model_id = f"{model_name}_lr{lr}_bs{batch_size}_dr{dropout_rate}_wd{weight_decay}_{optimizer_type}"
            best_val_loss = train_peta(
                peta_model,
                peta_train_loader,
                peta_val_loader,
                epochs=15,
                model_name=model_id,
                lr=lr,
                optimizer_type=optimizer_type,
                weight_decay=weight_decay,
                batch_size=batch_size,
                dropout_rate=dropout_rate,
                results_file=results_file
            )
            print(f"Results saved for {model_id}")

if __name__ == "__main__":
    main()