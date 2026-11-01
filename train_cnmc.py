import os
import csv
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import kagglehub
import torch.nn.functional as F
from tqdm import tqdm

# ==========================================
# 1. Configuration and Hyperparameters
# ==========================================
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.15
IMG_SIZE = 224
SEED = 42
NUM_WORKERS = 2
PATIENCE = 4
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

MODEL_NAME = "efficientnet_b0"

torch.manual_seed(SEED)
np.random.seed(SEED)

# ==========================================
# 2. Dataset Loading (Using KaggleHub)
# ==========================================
print("Downloading/Locating dataset via kagglehub...")
dataset_path = kagglehub.dataset_download("shafiullahshafin/c-nmc-2019-dataset")
print(f"Dataset downloaded to: {dataset_path}")

def find_class_dirs(root: Path) -> Path:
    """Find a directory that directly contains the 'all' and 'hem' folders."""
    root = root.resolve()
    candidates = [root]
    candidates.extend(path for path in root.rglob("*") if path.is_dir())
    for path in candidates:
        if (path / "all").is_dir() and (path / "hem").is_dir():
            return path
    raise FileNotFoundError(f"Could not find 'all' and 'hem' class folders under: {root}")

dataset_root = Path(dataset_path)
preferred_data_root = dataset_root / "C-NMC 2019 (PKG)" / "C-NMC_training_data" / "fold_0"
data_root = find_class_dirs(preferred_data_root if preferred_data_root.exists() else dataset_root)
print(f"Using image folder: {data_root}")

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

transform_val = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.14)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

dataset_full = datasets.ImageFolder(root=str(data_root), transform=None)
targets = np.array(dataset_full.targets)
splitter = StratifiedShuffleSplit(n_splits=1, test_size=VAL_SPLIT, random_state=SEED)
train_indices, val_indices = next(splitter.split(np.zeros_like(targets), targets))

train_dataset = Subset(
    datasets.ImageFolder(root=str(data_root), transform=transform_train), train_indices
)
val_dataset = Subset(
    datasets.ImageFolder(root=str(data_root), transform=transform_val), val_indices
)

train_targets = targets[train_indices]
class_sample_counts = np.bincount(train_targets, minlength=NUM_CLASSES)
class_weights = class_sample_counts.sum() / (
    NUM_CLASSES * np.maximum(class_sample_counts, 1)
)
class_weights = torch.tensor(class_weights, dtype=torch.float32)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True,
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True,
)

class_to_idx = dataset_full.class_to_idx
print(f"Classes: {class_to_idx}")
print(f"Class weights: {class_weights.tolist()}")

# ==========================================
# 3. Model Initialization
# ==========================================
print(f"Initializing {MODEL_NAME}...")
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
optimizer = torch.optim.AdamW(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=1
)
scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

# ==========================================
# 4. Training and Evaluation Loop
# ==========================================
def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            
            # Get probabilities for the positive class (class 1)
            probs = F.softmax(outputs, dim=1)[:, 1]
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    # Calculate Metrics
    val_loss = running_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    
    cm = confusion_matrix(all_labels, all_preds)
    # cm structure: [[TN, FP], [FN, TP]]
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # True Positive Rate / Recall
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0 # True Negative Rate
    else:
        sensitivity = 0.0
        specificity = 0.0
        
    return val_loss, acc, f1, auc, sensitivity, specificity

print("Starting training...")
csv_filename = f"{MODEL_NAME}_training_log.csv"
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        "Epoch", "Train Loss", "Val Loss", "Val Accuracy", "Val F1", "Val AUC",
        "Val Sensitivity", "Val Specificity", "Learning Rate",
    ])

best_val_loss = math.inf
best_epoch = -1

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")
    for i, (images, labels) in enumerate(train_pbar):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * images.size(0)
        
        # Update the progress bar suffix with the current loss
        train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
    # Evaluate at the end of each epoch
    print("Evaluating on validation set...")
    train_loss = running_loss / len(train_loader.dataset)
    val_loss, val_acc, val_f1, val_auc, val_sens, val_spec = evaluate(
        model, val_loader, criterion
    )
    scheduler.step(val_loss)
    
    print(f"--- Epoch {epoch+1} Results ---")
    print(f"Loss:        {train_loss:.4f}")
    print(f"Accuracy:    {val_acc:.4f}")
    print(f"F1 Score:    {val_f1:.4f}")
    print(f"AUC ROC:     {val_auc:.4f}")
    print(f"Sensitivity: {val_sens:.4f}")
    print(f"Specificity: {val_spec:.4f}")
    print(f"Val Loss:    {val_loss:.4f}")
    print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    print("-" * 25)
    
    # Log to CSV
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            epoch + 1, train_loss, val_loss, val_acc, val_f1, val_auc, val_sens,
            val_spec, optimizer.param_groups[0]["lr"],
        ])
        
    if val_loss < best_val_loss - 1e-5:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        torch.save(model.state_dict(), f"{MODEL_NAME}_best.pth")
        print(f"[*] New best model saved as {MODEL_NAME}_best.pth")
    elif epoch + 1 - best_epoch >= PATIENCE:
        print(f"Early stopping: no validation loss improvement for {PATIENCE} epochs.")
        break

print(
    f"Training complete! Best model saved from epoch {best_epoch} "
    f"with validation loss: {best_val_loss:.4f}"
)
