import pandas as pd
import glob
import os

print("Model Performance Summary (Best Val AUC):\n" + "="*50)
for f in glob.glob("*_training_log.csv"):
    model_name = f.replace('_training_log.csv', '')
    df = pd.read_csv(f)
    best_row = df.loc[df["Val AUC"].idxmax()]
    print(f"Model: {model_name}")
    print(f"  - Epoch:       {int(best_row['Epoch'])}")
    print(f"  - Accuracy:    {best_row['Val Accuracy']:.4f}")
    print(f"  - AUC ROC:     {best_row['Val AUC']:.4f}")
    print(f"  - F1 Score:    {best_row['Val F1']:.4f}")
    print(f"  - Sensitivity: {best_row['Val Sensitivity']:.4f}")
    print(f"  - Specificity: {best_row['Val Specificity']:.4f}")
    print("-" * 30)
