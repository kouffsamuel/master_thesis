import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('/Benson_DATA3/Public/RADIal/ready_to_use/RADIal/labels_with_class.csv')

df_labeled = df[df['x1_pix'] != -1]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0,0].hist(df_labeled['conf_class'], bins=50, color='coral', edgecolor='black')
axes[0,0].axvline(x=0.5, color='red', linestyle='--', label='0.5')
axes[0,0].axvline(x=0.7, color='green', linestyle='--', label='0.7')
axes[0,0].set_title('Distribution of Classification Confidence')
axes[0,0].set_xlabel('Confidence')
axes[0,0].legend()

axes[0,1].hist(df_labeled['iou_class'], bins=50, color='steelblue', edgecolor='black')
axes[0,1].axvline(0.3, color='red',    linestyle='--', label='0.3')
axes[0,1].axvline(0.6, color='green',  linestyle='--', label='0.6')
axes[0,1].set_title('Distribution IoU')
axes[0,1].set_xlabel('IoU')
axes[0,1].legend()

scatter = axes[1,0].scatter(
    df_labeled['iou_class'],
    df_labeled['conf_class'],
    c=df_labeled['vehicle_class'].astype('category').cat.codes,
    alpha=0.3, s=5, cmap='tab10'
)
axes[1,0].axvline(0.6, color='green', linestyle='--', label='IoU=0.6')
axes[1,0].axhline(0.5, color='red',   linestyle='--', label='conf=0.5')
axes[1,0].set_xlabel('IoU')
axes[1,0].set_ylabel('Confidence')
axes[1,0].set_title('IoU vs Confidence per class')
axes[1,0].legend()

df_labeled.groupby('vehicle_class')['conf_class'].mean().plot(
    kind='bar', ax=axes[1,1], color='coral')
axes[1,1].axhline(0.5, color='red', linestyle='--', label='seuil 0.5')
axes[1,1].set_title('Mean confidence per class')
axes[1,1].legend()

plt.tight_layout()
plt.savefig('/home/skouff/master_thesis/dataset/classification_distribution_corrected_3.png')

class_counts = df_labeled['vehicle_class'].value_counts()

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(class_counts.index, class_counts.values, color='steelblue', edgecolor='black')

# Afficher le count au dessus de chaque barre
for bar, count in zip(bars, class_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
            str(count), ha='center', va='bottom', fontsize=11)

ax.set_title('Number of samples per class')
ax.set_xlabel('Class')
ax.set_ylabel('Number of samples')
plt.tight_layout()
plt.savefig('/home/skouff/master_thesis/dataset/class_counts_corrected_3.png')

print(class_counts)
print(f"\nTotal samples labelisés : {class_counts.sum()}")

# high  = (df_labeled['iou_class'] > 0.6) & (df_labeled['conf_class'] > 0.5)
# mid   = ~high & ((df_labeled['iou_class'] >= 0.3) | (df_labeled['conf_class'] >= 0.5))
# low   = (df_labeled['iou_class'] < 0.3) & (df_labeled['conf_class'] < 0.5)

# print(f"IoU>0.6 ET conf>0.5 (reliable)      : {high.sum()}")
# print(f"Mixte (review)               : {mid.sum()}")
# print(f"IoU<0.3 ET conf<0.5 (modify)   : {low.sum()}")