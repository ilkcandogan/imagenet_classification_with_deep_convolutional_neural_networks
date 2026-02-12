import os
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from tqdm import tqdm


def build_dataloader(dataset, batch_size, shuffle, num_workers, pin_memory):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )


class ComofodDataset(Dataset):
    def __init__(self, images_dir, mask_dir, transform_image, transform_mask):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        self.transform_image = transform_image
        self.transform_mask = transform_mask
        self.image_files = sorted([p.name for p in self.images_dir.iterdir() if p.is_file()])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_name = self.image_files[index]
        image_path = self.images_dir / image_name

        mask_name = image_name.split("_")[0] + "_M.png"
        mask_path = self.mask_dir / mask_name

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.transform_image(image)
        mask = self.transform_mask(mask)
        mask = (mask > 0).float()

        return image, mask


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = device.type == "cuda"
pin_memory = device.type == "cuda"
print(f"Training on {device} | AMP={use_amp}")

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

batch_size = int(os.getenv("BATCH_SIZE", "16"))
num_workers = int(os.getenv("NUM_WORKERS", "4"))
num_epochs = int(os.getenv("NUM_EPOCHS", "1"))
seed = int(os.getenv("SEED", "42"))

transform_image = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

transform_mask = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])

full_dataset = ComofodDataset(
    "/app/dataset/processed/task2/images",
    "/app/dataset/processed/task2/masks",
    transform_image,
    transform_mask,
)

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
split_generator = torch.Generator().manual_seed(seed)
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size], generator=split_generator)

val_size = test_size // 2
test_size = test_size - val_size
test_dataset, val_dataset = random_split(test_dataset, [test_size, val_size], generator=split_generator)

train_dataloader = build_dataloader(train_dataset, batch_size, True, num_workers, pin_memory)
val_dataloader = build_dataloader(val_dataset, batch_size, False, num_workers, pin_memory)

model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
).to(device)
if device.type == "cuda":
    model = model.to(memory_format=torch.channels_last)

optimizer = torch.optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

best_val_loss = float("inf")

save_dir = Path("/app/outputs/task2/")
save_dir.mkdir(exist_ok=True)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_loader_tqdm = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]", leave=False)

    for images, masks in train_loader_tqdm:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if device.type == "cuda":
            images = images.to(memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        train_loader_tqdm.set_postfix({"loss": f"{loss.item():.4f}"})

    train_loss /= len(train_dataloader)

    model.eval()
    val_loss = 0.0
    val_loader_tqdm = tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]", leave=False)

    with torch.no_grad():
        for images, masks in val_loader_tqdm:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            if device.type == "cuda":
                images = images.to(memory_format=torch.channels_last)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, masks)

            val_loss += loss.item()
            val_loader_tqdm.set_postfix({"val_loss": f"{loss.item():.4f}"})

    val_loss /= len(val_dataloader)

    print(f"Epoch {epoch + 1}/{num_epochs} | " f"Train Loss: {train_loss:.4f} | " f"Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), save_dir / "best_unet_model.pth")
        print(f"Best model saved: {save_dir / 'best_unet_model.pth'}")
