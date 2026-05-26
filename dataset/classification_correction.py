import pandas as pd 
import cv2
import numpy as np
import sys
import os 
sys.path.append('/home/skouff/')
sys.path.append('/home/skouff/master_thesis/')
from RADIal.DBReader import SyncReader

CLASS_KEYS = {ord('c'): 'car', ord('t'): 'truck', ord('b'): 'bus', ord('p'): 'person'}

df = pd.read_csv('/Benson_DATA3/Public/RADIal/ready_to_use/RADIal/labels_with_class.csv')
results_df = df.copy()
df_filter = df[(df['vehicle_class'] == 'truck') | (df['conf_class'] < 0.5)]

db_cache = {}
for idx, row in df_filter.iterrows():
    if row['dataset'] not in db_cache:
        db = SyncReader(os.path.join('/Benson_DATA3/Public/RADIal/raw_sequences/', row['dataset']), tolerance=20000, silent=True)
        db_cache[row['dataset']] = db
    db = db_cache[row['dataset']]
    img = db.GetSensorData(row['index'])['camera']['data'].copy()
    x1, y1, x2, y2 = int(row['x1_pix']), int(row['y1_pix']), int(row['x2_pix']), int(row['y2_pix'])
    
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f"{row['vehicle_class']} (conf: {row['conf_class']:.2f}, IoU: {row['iou_class']:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    pad = 80
    crop = img[max(0,y1-pad):y2+pad, max(0,x1-pad):x2+pad]
    cv2.imshow('[c]ar [t]ruck [p]erson [b]us', crop)

    key = cv2.waitKey(0) & 0xFF
    # Ajouter 'q' pour quitter et sauvegarder
    results_df.loc[idx, 'vehicle_class'] = CLASS_KEYS.get(key, row['vehicle_class'])
    results_df.loc[idx, 'conf_class'] = 1.0 if key in CLASS_KEYS else row['conf_class']
    if key == ord('q'):
        print("Sauvegarde intermédiaire...")
        results_df.to_csv('/Benson_DATA3/Public/RADIal/ready_to_use/RADIal/labels_with_class_corrected.csv', index=False)
        break
    
cv2.destroyAllWindows()
results_df.to_csv('/Benson_DATA3/Public/RADIal/ready_to_use/RADIal/labels_with_class_corrected.csv', index=False)