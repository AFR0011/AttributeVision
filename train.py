"""Clean training/evaluation entry point for the Biometrics Application.

The original ``training.py`` is retained as the historical experiment and
hyperparameter-search script. This entry point fixes two issues that matter
for a clean reproducible run:

1. PETA models emit logits and are optimized with BCEWithLogitsLoss.
2. Training and validation subsets use distinct transform pipelines.
"""

from __future__ import annotations

import argparse
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset, default_collate

from training import PETADataset, UTKFaceDataset, get_transforms


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_skip_invalid(batch):
    valid = [item for item in batch if item is not None]
    return default_collate(valid) if valid else None


def split_with_separate_transforms(dataset: str, root: str, seed: int, train_fraction: float = 0.8):
    if dataset == "peta":
        train_base = PETADataset(root, transform=get_transforms("PETA", True))
        val_base = PETADataset(root, transform=get_transforms("PETA", False))
    else:
        train_base = UTKFaceDataset(root, transform=get_transforms("UTKFace", True))
        val_base = UTKFaceDataset(root, transform=get_transforms("UTKFace", False))

    if len(train_base) == 0:
        raise ValueError(f"No samples found under {root}")
    if len(train_base) != len(val_base):
        raise ValueError("Training and validation dataset views do not align.")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_base), generator=generator).tolist()
    cut = int(train_fraction * len(indices))
    train_idx, val_idx = indices[:cut], indices[cut:]
    return Subset(train_base, train_idx), Subset(val_base, val_idx)


class MobileNetMultiTask(nn.Module):
    def __init__(self, dataset: str, dropout: float = 0.3):
        super().__init__()
        self.dataset = dataset
        self.backbone = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
        self.backbone.classifier = nn.Identity()

        # Fine-tune only the last three feature blocks.
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for block in self.backbone.features[-3:]:
            for parameter in block.parameters():
                parameter.requires_grad = True

        with torch.no_grad():
            features = self.backbone(torch.zeros(1, 3, 224, 224))
            in_features = features.shape[1]

        self.shared = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        if dataset == "utkface":
            self.age_head = nn.Linear(512, 1)
            self.gender_head = nn.Linear(512, 2)
            self.race_head = nn.Linear(512, 5)
        else:
            self.attribute_head = nn.Linear(512, 105)

    def forward(self, x: torch.Tensor):
        features = self.shared(self.backbone(x))
        if self.dataset == "utkface":
            return {
                "age": self.age_head(features).squeeze(-1),
                "gender": self.gender_head(features),
                "race": self.race_head(features),
            }
        return self.attribute_head(features)


class CustomCNN(nn.Module):
    def __init__(self, dataset: str, dropout: float = 0.3, conv_layers: int = 3, hidden_units: int = 512):
        super().__init__()
        self.dataset = dataset
        channels = [64, 128, 256, 512][:conv_layers]
        layers: list[nn.Module] = []
        in_channels = 3
        for out_channels in channels:
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                ]
            )
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * 4 * 4, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        if dataset == "utkface":
            self.age_head = nn.Linear(hidden_units, 1)
            self.gender_head = nn.Linear(hidden_units, 2)
            self.race_head = nn.Linear(hidden_units, 5)
        else:
            self.attribute_head = nn.Linear(hidden_units, 105)

    def forward(self, x: torch.Tensor):
        features = self.shared(self.pool(self.features(x)))
        if self.dataset == "utkface":
            return {
                "age": self.age_head(features).squeeze(-1),
                "gender": self.gender_head(features),
                "race": self.race_head(features),
            }
        return self.attribute_head(features)


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def train_epoch(model, loader, optimizer, device, dataset, scaler):
    model.train()
    total_loss = 0.0
    batches = 0
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    for batch in loader:
        if batch is None:
            continue
        images, labels = batch
        images = images.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device):
            outputs = model(images)
            if dataset == "peta":
                labels = labels.to(device, non_blocking=True)
                loss = bce(outputs, labels)
            else:
                labels = {k: v.to(device, non_blocking=True) for k, v in labels.items()}
                age_loss = mse(outputs["age"], labels["age"])
                gender_loss = ce(outputs["gender"], labels["gender"])
                race_loss = ce(outputs["race"], labels["race"])
                loss = 0.1 * age_loss + gender_loss + race_loss

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.detach().cpu())
        batches += 1
    return total_loss / max(batches, 1)


@torch.no_grad()
def evaluate(model, loader, device, dataset) -> Dict[str, float]:
    model.eval()
    if dataset == "peta":
        all_preds, all_labels = [], []
        for batch in loader:
            if batch is None:
                continue
            images, labels = batch
            logits = model(images.to(device, non_blocking=True))
            preds = (torch.sigmoid(logits) >= 0.5).int().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.numpy())
        return {
            "micro_f1": float(
                f1_score(np.concatenate(all_labels), np.concatenate(all_preds), average="micro")
            )
        }

    age_errors, gender_ok, race_ok, count = [], 0, 0, 0
    for batch in loader:
        if batch is None:
            continue
        images, labels = batch
        outputs = model(images.to(device, non_blocking=True))
        age_true = labels["age"].to(device)
        gender_true = labels["gender"].to(device)
        race_true = labels["race"].to(device)
        age_errors.extend((torch.abs(outputs["age"] - age_true) * 116.0).cpu().tolist())
        gender_ok += int((outputs["gender"].argmax(1) == gender_true).sum())
        race_ok += int((outputs["race"].argmax(1) == race_true).sum())
        count += len(images)
    return {
        "age_mae": float(np.mean(age_errors)),
        "gender_accuracy": gender_ok / max(count, 1),
        "race_accuracy": race_ok / max(count, 1),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train the multi-attribute biometrics models.")
    parser.add_argument("--dataset", choices=["utkface", "peta"], required=True)
    parser.add_argument("--model", choices=["mobilenet", "mlcnn"], default="mobilenet")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set, val_set = split_with_separate_transforms(args.dataset, args.data_root, args.seed)
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_skip_invalid,
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)

    if args.model == "mobilenet":
        model = MobileNetMultiTask(args.dataset, args.dropout)
    else:
        model = CustomCNN(args.dataset, args.dropout)
    model.to(device)

    optimizer = optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_score = float("inf") if args.dataset == "utkface" else float("-inf")
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)
    checkpoint = output_dir / f"best_{args.dataset}_{args.model}.pt"

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, device, args.dataset, scaler)
        metrics = evaluate(model, val_loader, device, args.dataset)
        print(f"epoch={epoch:03d} loss={loss:.5f} metrics={metrics}")

        score = metrics["age_mae"] if args.dataset == "utkface" else metrics["micro_f1"]
        improved = score < best_score if args.dataset == "utkface" else score > best_score
        if improved:
            best_score = score
            torch.save(model.state_dict(), checkpoint)

    print(f"Saved best checkpoint to {checkpoint}")


if __name__ == "__main__":
    main()
