import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set premium aesthetic style
plt.style.use('dark_background')
sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#121212", "figure.facecolor": "#121212", "grid.color": "#2c2c2c"})

# 1. Gather Data
summary_data = []
all_histories = {}

files = glob.glob("*_training_log.csv")
for f in files:
    model_name = f.replace('_training_log.csv', '')
    # Clean up long names for plotting
    if model_name == 'swin_tiny_patch4_window7_224':
        display_name = 'Swin-Tiny'
    elif model_name == 'efficientnetv2_rw_s':
        display_name = 'EffNetV2-S'
    elif model_name == 'efficientnet_b0':
        display_name = 'EffNet-B0'
    elif model_name == 'convnext_tiny':
        display_name = 'ConvNeXt-Tiny'
    elif model_name == 'resnet50':
        display_name = 'ResNet-50'
    else:
        display_name = model_name

    df = pd.read_csv(f)
    all_histories[display_name] = df
    
    best_idx = df["Val AUC"].idxmax()
    best_row = df.loc[best_idx].copy()
    best_row['Model'] = display_name
    summary_data.append(best_row)

# Create and save the summary CSV
summary_df = pd.DataFrame(summary_data)
# Reorder columns
cols = ['Model', 'Epoch', 'Val AUC', 'Val Accuracy', 'Val F1', 'Val Sensitivity', 'Val Specificity', 'Train Loss']
summary_df = summary_df[cols].sort_values(by='Val AUC', ascending=False).reset_index(drop=True)
summary_df.to_csv("all_models_summary.csv", index=False)
print("Saved summary to all_models_summary.csv")

# 2. Create Stunning Charts
fig = plt.figure(figsize=(16, 8), dpi=200)

# Color palette
colors = sns.color_palette("cool", len(summary_df))

# Plot A: Bar Chart of Best AUC
ax1 = plt.subplot(1, 2, 1)
bars = sns.barplot(x='Val AUC', y='Model', data=summary_df, palette="cool", ax=ax1, edgecolor='#ffffff', linewidth=1.5)
ax1.set_title('🏆 Peak Validation AUC by Model', fontsize=18, fontweight='bold', color='white', pad=20)
ax1.set_xlabel('Validation AUC Score', fontsize=14, color='lightgray')
ax1.set_ylabel('')
ax1.set_xlim(0.9, 1.0) # Zoom in to see the differences clearly
ax1.tick_params(colors='lightgray', labelsize=12)

# Add data labels to bars
for i, bar in enumerate(ax1.containers[0]):
    width = bar.get_width()
    ax1.text(width + 0.002, bar.get_y() + bar.get_height()/2., 
             f'{width:.4f}', 
             ha='left', va='center', color='white', fontweight='bold', fontsize=12)

# Plot B: Learning Curves (AUC over Epochs)
ax2 = plt.subplot(1, 2, 2)
for i, (name, df) in enumerate(all_histories.items()):
    sns.lineplot(x='Epoch', y='Val AUC', data=df, label=name, ax=ax2, linewidth=3, marker='o', markersize=6)

ax2.set_title('📈 AUC Progression Over Time', fontsize=18, fontweight='bold', color='white', pad=20)
ax2.set_xlabel('Training Epoch', fontsize=14, color='lightgray')
ax2.set_ylabel('Validation AUC', fontsize=14, color='lightgray')
ax2.tick_params(colors='lightgray', labelsize=12)

# Premium legend
leg = ax2.legend(loc='lower right', frameon=True, fontsize=11, title="Architectures")
leg.get_frame().set_facecolor('#1e1e1e')
leg.get_frame().set_edgecolor('#444444')
plt.setp(leg.get_title(), color='white', fontweight='bold')
for text in leg.get_texts():
    text.set_color("lightgray")

plt.tight_layout(pad=3.0)
plt.savefig('model_comparison.png', facecolor='#121212', bbox_inches='tight')
print("Saved charts to model_comparison.png")
