import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import kagglehub
import timm  # Make sure to run: pip install timm scikit-learn
import torch.nn.functional as F
from tqdm import tqdm

# ==========================================
# 1. Configuration and Hyperparameters
# ==========================================
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Select your model here from the list below
# Options: 'efficientnet_b0', 'efficientnetv2_rw_s', 'convnext_tiny', 
#          'swin_tiny_patch4_window7_224', 'densenet121', 'resnet50'
MODEL_NAME = 'efficientnet_b0' 

# ==========================================
# 2. Dataset Loading (Using KaggleHub)
# ==========================================
print("Downloading/Locating dataset via kagglehub...")
# Note: The Pandas adapter you provided is for CSVs. 
# For images, we just download the path and use PyTorch's ImageFolder.
dataset_path = kagglehub.dataset_download("shafiullahshafin/c-nmc-2019-dataset")
print(f"Dataset downloaded to: {dataset_path}")

# The C-NMC dataset has multiple folds for training. We'll use fold_0 for demonstration.
# Path structure: C-NMC_training_data/fold_0/all  and C-NMC_training_data/fold_0/hem
train_dir = os.path.join(dataset_path, "C-NMC 2019 (PKG)", "C-NMC_training_data", "fold_0")
val_dir = os.path.join(dataset_path, "C-NMC 2019 (PKG)", "C-NMC_training_data", "fold_1") # Using fold 1 as validation

# Define Data Augmentations and Transforms
# Models like Swin/ConvNeXt expect 224x224 inputs generally.
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print(f"Loading images from {train_dir}...")
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform_train)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

print(f"Classes: {train_dataset.classes} (0: {train_dataset.classes[0]}, 1: {train_dataset.classes[1]})")

# ==========================================
# 3. Model Initialization
# ==========================================
print(f"Initializing {MODEL_NAME}...")
# We use the 'timm' (PyTorch Image Models) library to easily load any of these models.
model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=2)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# ==========================================
# 4. Training and Evaluation Loop
# ==========================================
def evaluate(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            
            # Get probabilities for the positive class (class 1)
            probs = F.softmax(outputs, dim=1)[:, 1]
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    # Calculate Metrics
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
        
    return acc, f1, auc, sensitivity, specificity

print("Starting training...")
csv_filename = f"{MODEL_NAME}_training_log.csv"
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Train Loss", "Val Accuracy", "Val F1", "Val AUC", "Val Sensitivity", "Val Specificity"])

best_val_auc = 0.0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")
    for i, (images, labels) in enumerate(train_pbar):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Update the progress bar suffix with the current loss
        train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
    # Evaluate at the end of each epoch
    print("Evaluating on validation set...")
    val_acc, val_f1, val_auc, val_sens, val_spec = evaluate(model, val_loader)
    
    print(f"--- Epoch {epoch+1} Results ---")
    print(f"Loss:        {running_loss/len(train_loader):.4f}")
    print(f"Accuracy:    {val_acc:.4f}")
    print(f"F1 Score:    {val_f1:.4f}")
    print(f"AUC ROC:     {val_auc:.4f}")
    print(f"Sensitivity: {val_sens:.4f}")
    print(f"Specificity: {val_spec:.4f}")
    print("-" * 25)
    
    # Log to CSV
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch+1, running_loss/len(train_loader), val_acc, val_f1, val_auc, val_sens, val_spec])
        
    # Save the best model
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save(model.state_dict(), f"{MODEL_NAME}_best.pth")
        print(f"[*] New best model saved as {MODEL_NAME}_best.pth")

print(f"Training complete! Best model saved with AUC: {best_val_auc:.4f}")
