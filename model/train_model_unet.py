import os
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# Класс датасета
class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        # Фильтруем: берем только файлы (исключаем папки и скрытые файлы)
        self.images = [
            f for f in os.listdir(images_dir)
            if os.path.isfile(os.path.join(images_dir, f)) and not f.startswith('.')
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.images[idx].replace(".jpg", "_mask.png"))  # подкорректируй под свои имена

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))  # Грейскейл маска

        # Нормализуем маску к 0/1
        mask = (mask > 128).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask.unsqueeze(0)  # маска с каналом (1, H, W)
    
# Метрика Dice 
def dice_coef(y_pred, y_true, smooth=1e-6):
    y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred > 0.5).float()
    intersection = (y_pred * y_true).sum(dim=(1,2,3))
    union = y_pred.sum(dim=(1,2,3)) + y_true.sum(dim=(1,2,3))
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice.mean().item()

# Функция обучения на эпоху 
def train_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0
    running_dice = 0
    for images, masks in tqdm(loader):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = dice_loss_fn(outputs, masks) + bce_loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_dice += dice_coef(outputs, masks)

    return running_loss / len(loader), running_dice / len(loader)

# Функция валидации 
def eval_epoch(model, loader, device):
    model.eval()
    running_loss = 0
    running_dice = 0
    with torch.no_grad():
        for images, masks in tqdm(loader):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = dice_loss_fn(outputs, masks) + bce_loss_fn(outputs, masks)

            running_loss += loss.item()
            running_dice += dice_coef(outputs, masks)

    return running_loss / len(loader), running_dice / len(loader)

# Настройки
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
IMAGE_SIZE = 512
EPOCHS = 10
TRAIN_IMG_DIR = "/satellite-plot-segmentation-1/train"
TRAIN_MASK_DIR = "/satellite-plot-segmentation-1/train_masks"
VAL_IMG_DIR = "/satellite-plot-segmentation-1/valid"
VAL_MASK_DIR = "/satellite-plot-segmentation-1/valid_masks"

# Трансформация и аугментация датасета
train_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ColorJitter(p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Загрузка датасета
train_ds = SegmentationDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, transform=train_transform)
val_ds = SegmentationDataset(VAL_IMG_DIR, VAL_MASK_DIR, transform=val_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# Модель 
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
).to(DEVICE)

# Функции потерь
dice_loss_fn = smp.losses.DiceLoss(mode="binary")
bce_loss_fn = nn.BCEWithLogitsLoss()

# Оптимизатор
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

train_losses, val_losses = [], []
train_dices, val_dices = [], []

# Обучение
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    train_loss, train_dice = train_epoch(model, train_loader, optimizer, DEVICE)
    val_loss, val_dice = eval_epoch(model, val_loader, DEVICE)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_dices.append(train_dice)
    val_dices.append(val_dice)

    print(f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}")
    print(f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")

