import os
import csv
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Subset
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

def find_training_root(root: Path) -> Path:
    """Find the directory containing all three official, patient-separated folds."""
    root = root.resolve()
    fold_zero_dirs = [path for path in root.rglob("fold_0") if path.is_dir()]
    for fold_zero in fold_zero_dirs:
        training_root = fold_zero.parent
        fold_dirs = [training_root / f"fold_{index}" for index in range(3)]
        if all(
            (fold / "all").is_dir() and (fold / "hem").is_dir()
            for fold in fold_dirs
        ):
            return training_root
    raise FileNotFoundError(
        f"Could not find fold_0, fold_1, and fold_2 with 'all'/'hem' folders under: {root}"
    )


dataset_root = Path(dataset_path)
training_root = find_training_root(dataset_root)
fold_roots = [training_root / f"fold_{fold}" for fold in range(3)]
print(f"Using notebook-style train/validation split across: {training_root}")

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

index_datasets = [
    datasets.ImageFolder(root=str(fold_root), transform=None)
    for fold_root in fold_roots
]
class_to_idx = index_datasets[0].class_to_idx
if any(dataset.class_to_idx != class_to_idx for dataset in index_datasets[1:]):
    raise ValueError("Class folders differ between the official dataset folds.")
targets = np.concatenate(
    [np.asarray(dataset.targets) for dataset in index_datasets]
)
splitter = StratifiedShuffleSplit(
    n_splits=1, test_size=VAL_SPLIT, random_state=SEED
)
train_indices, val_indices = next(
    splitter.split(np.zeros_like(targets), targets)
)
train_dataset = Subset(
    ConcatDataset([
        datasets.ImageFolder(root=str(fold_root), transform=transform_train)
        for fold_root in fold_roots
    ]),
    train_indices,
)
val_dataset = Subset(
    ConcatDataset([
        datasets.ImageFolder(root=str(fold_root), transform=transform_val)
        for fold_root in fold_roots
    ]),
    val_indices,
)
if class_to_idx.get("all") != 0 or class_to_idx.get("hem") != 1:
    raise ValueError(
        f"Expected ImageFolder class mapping all=0, hem=1; got {class_to_idx}"
    )
positive_class = class_to_idx["all"]

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

print(f"Classes: {class_to_idx}")
print(f"Class mapping: {class_to_idx} (positive/cancer class: all)")
print(f"Training images: {len(train_dataset)}; validation images: {len(val_dataset)}")
print(f"Class weights: {class_weights.tolist()}")

# ==========================================
# 3. Model Initialization
# ==========================================
print(f"Initializing {MODEL_NAME}...")
if MODEL_NAME == "efficientnet_b0":
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
elif MODEL_NAME == "convnext_tiny":
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, NUM_CLASSES)
elif MODEL_NAME == "resnet50":
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
elif MODEL_NAME == "densenet121":
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
elif MODEL_NAME == "swin_tiny_patch4_window7_224":
    model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
    model.head = nn.Linear(model.head.in_features, NUM_CLASSES)
else:
    raise ValueError(f"Unsupported MODEL_NAME: {MODEL_NAME}")
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
            
            probs = F.softmax(outputs, dim=1)[:, positive_class]
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    # Calculate Metrics
    val_loss = running_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    positive_labels = np.asarray(all_labels) == positive_class
    f1 = f1_score(all_labels, all_preds, average="weighted")
    auc = roc_auc_score(positive_labels, all_probs)
    
    cm = confusion_matrix(
        all_labels, all_preds, labels=[1 - positive_class, positive_class]
    )
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
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
