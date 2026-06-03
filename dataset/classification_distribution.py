import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

"""
Script for visualizing the distribution of classes in the dataset, based on the labels_with_class.csv file. 
This file contains the corrected labels after manual review, and this script will help us understand the class imbalance in our dataset. 
The resulting bar chart will show the number of samples for each class (car, truck, bus, bicycle, person) and will be saved as 'class_counts_corrected_3.png'.
Written with help of Claude.ai
"""

df = pd.read_csv('/Benson_DATA3/Public/RADIal/ready_to_use/RADIal/labels_with_class.csv')

df_labeled = df[df['x1_pix'] != -1]

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