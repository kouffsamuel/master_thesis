from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np
import sys
from tqdm import tqdm

from shapely import Polygon 
sys.path.append('/home/skouff/')
sys.path.append('/home/skouff/master_thesis/')
from RADIal.DBReader import SyncReader
from ultralytics import YOLO
import os

def bbox_iou(box1, boxes):
    x1, y1, x2, y2 = box1

    xi1 = np.maximum(x1, boxes[:, 0])
    yi1 = np.maximum(y1, boxes[:, 1])
    xi2 = np.minimum(x2, boxes[:, 2])
    yi2 = np.minimum(y2, boxes[:, 3])

    inter_w = np.maximum(0, xi2 - xi1)
    inter_h = np.maximum(0, yi2 - yi1)
    inter_area = inter_w * inter_h

    area1 = (x2 - x1) * (y2 - y1)
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    union = area1 + area2 - inter_area

    return np.where(union > 0, inter_area / union, 0)



labels_df = pd.read_csv('/Benson_DATA3/Public/RADIal/ready_to_use/RADIal/labels.csv').to_numpy()
model = YOLO('yolo26x.pt')
vehicle_class_col = np.full(len(labels_df), -1, dtype=str)
class_tresh_col = np.full(len(labels_df), -1, dtype=float)

db_cache = {}
pbar = tqdm(enumerate(labels_df), total=len(labels_df), desc="Classifying", unit="sample")

for idx, label in pbar:
    if label[1] == -1 : 
        continue

    if label[14] not in db_cache:
        db = SyncReader(os.path.join('/Benson_DATA3/Public/RADIal/raw_sequences/',label[14]), tolerance=20000, silent=True)
        db_cache[label[14]] = db
    
    db = db_cache[label[14]]

    data = db.GetSensorData(label[15])
    # Get the camera image
    camera_data = data['camera']['data']
    # ax.imshow(camera_data)
    img_h, img_w = camera_data.shape[:2]
    
    result = model(camera_data, verbose=False)[0]
    conf_np = result.boxes.conf.cpu().numpy()
    cls_np = result.boxes.cls.cpu().numpy().astype(int)
    xyxy_np = result.boxes.xyxy.cpu().numpy()

    if len(xyxy_np) == 0:
        vehicle_class_col[idx] = 'nothing'
        class_tresh_col[idx] = 0.0
        pbar.set_postfix({
            'seq': label[14][-10:],
            'iou': f"{0.0:.2f}",
            'cls': 'nothing',
            'labeled': int(np.sum(vehicle_class_col != '-1'))
        })
        continue

    label_box= label[1:5].astype(np.float32)

    ious = bbox_iou(label_box, xyxy_np)
    best = int(np.argmax(ious))

    class_pred_id = cls_np[best]
    class_pred_name = result.names[class_pred_id]
    xyxy_np_best = xyxy_np[best]

    if ious[best] > 0.6:
        vehicle_class_col[idx] = class_pred_name
    
    class_tresh_col[idx] = ious[best]

    pbar.set_postfix({
        'seq': label[14][-10:],
        'iou': f"{ious[best]:.2f}",
        'cls': class_pred_name,
        'labeled': int(np.sum(vehicle_class_col != '-1'))
    })

labels_df['vehicle_class'] = vehicle_class_col
labels_df['iou_class'] = class_tresh_col
labels_df.to_csv('/Benson_DATA3/Public/RADIal/ready_to_use/RADIal/labels_with_class.csv', index=False)
