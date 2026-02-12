import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm


def build_dataloader(dataset, batch_size, shuffle, num_workers, pin_memory):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = device.type == "cuda"
pin_memory = device.type == "cuda"
print(f"Training on {device} | AMP={use_amp}")

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

batch_size = int(os.getenv("BATCH_SIZE", "32"))
num_workers = int(os.getenv("NUM_WORKERS", "4"))
num_epochs = int(os.getenv("NUM_EPOCHS", "1"))

train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_data = datasets.ImageFolder("/app/dataset/processed/task1/train", transform=train_transform)
val_data = datasets.ImageFolder("/app/dataset/processed/task1/val", transform=val_transform)

train_loader = build_dataloader(train_data, batch_size, True, num_workers, pin_memory)
val_loader = build_dataloader(val_data, batch_size, False, num_workers, pin_memory)

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 1)
model = model.to(device)
if device.type == "cuda":
    model = model.to(memory_format=torch.channels_last)

optimizer = torch.optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

best_val_acc = 0.0

save_dir = Path("/app/outputs/task1/")
save_dir.mkdir(exist_ok=True)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]", leave=False)

    for images, labels in train_loader_tqdm:
        images = images.to(device, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(device, non_blocking=True)
        if device.type == "cuda":
            images = images.to(memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        train_loader_tqdm.set_postfix({"loss": f"{loss.item():.4f}"})

    train_loss = running_loss / len(train_loader)

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]", leave=False)

    with torch.no_grad():
        for images, labels in val_loader_tqdm:
            images = images.to(device, non_blocking=True)
            labels = labels.float().unsqueeze(1).to(device, non_blocking=True)
            if device.type == "cuda":
                images = images.to(memory_format=torch.channels_last)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            val_loss += loss.item()
            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            val_loader_tqdm.set_postfix({"val_loss": f"{loss.item():.4f}"})

    val_loss /= len(val_loader)
    val_acc = correct / total

    print(
        f"Epoch {epoch + 1}/{num_epochs} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), save_dir / "best_resnet18_model.pth")
        print(f"Best model saved: {save_dir / 'best_resnet18_model.pth'}")
