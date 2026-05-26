import argparse
import json
import os
import random
import sys

from matplotlib import image
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import torch
import pandas as pd

from dataset.encoder import ra_encoder
from dataset.dataloader import CreateDataLoaders
from model.RadViT import RadViT
from dataset.dataset import RADIal
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
sys.path.append('/home/skouff/')
sys.path.append('/home/skouff/master_thesis/')
from RADIal.DBReader import SyncReader

SEQUENCES = {'Test': [
    'RECORD@2020-11-22_12.45.05',
    'RECORD@2020-11-22_12.25.47',
    'RECORD@2020-11-22_12.03.47',
    'RECORD@2020-11-22_12.54.38'
]}

CLASS_MAP = {
    '0': 'car',
    '1': 'truck',
    '2': 'bicycle',
    '3': 'bus',
    '4': 'person',
}
NAME_TO_ID = {v: k for k, v in CLASS_MAP.items()}
CLASS_NAMES = list(CLASS_MAP.values())

def bbox_iou(box1, boxes):
    x1, y1, x2, y2 = box1
    xi1 = np.maximum(x1, boxes[:, 0]); yi1 = np.maximum(y1, boxes[:, 1])
    xi2 = np.minimum(x2, boxes[:, 2]); yi2 = np.minimum(y2, boxes[:, 3])
    inter = np.maximum(0, xi2 - xi1) * np.maximum(0, yi2 - yi1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - inter
    return np.where(union > 0, inter / union, 0)


def main(config):
    # Setup random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load the dataset
    
    labels = pd.read_csv(os.path.join(config['dataset']['root_dir'],'labels_with_class.csv')).to_numpy()

    # Build cache for test sequences and collect test data
    db_cache = {}
    test_set = []

    for seq in SEQUENCES['Test']:
        db_cache[seq] = SyncReader(
            os.path.join('/Benson_DATA3/Public/RADIal/raw_sequences/', seq), 
            tolerance=20000, 
            silent=True
        )
        idx = np.where(labels[:, 14] == seq)[0]
        test_set.append(labels[idx,0])

    model_26 = YOLO('/home/skouff/master_thesis/dataset/yolo26x.pt')

    unique_sample_ids = np.unique(np.concatenate(test_set))

    all_pred_cls = []
    all_gt_cls   = []
    no_match_count = 0

    for sample_id in unique_sample_ids:
        # Toutes les lignes de labels pour ce sample
        sample_rows = np.where(labels[:, 0] == sample_id)[0]
        labels_for_image = labels[sample_rows]

        if len(labels_for_image) == 0:
            continue

        # frame_idx et seq sont les mêmes pour toutes les lignes du sample
        frame_idx = labels_for_image[0, 15]
        seq       = labels_for_image[0, 14]
        
        # Load image once
        
        data = db_cache[seq].GetSensorData(frame_idx)
        img = data['camera']['data']

        result  = model_26(img, verbose=False)[0]
        conf_np = result.boxes.conf.cpu().numpy()
        cls_np  = result.boxes.cls.cpu().numpy().astype(int)
        xyxy_np = result.boxes.xyxy.cpu().numpy()
        # Draw all boxes for this image
        for label in labels_for_image:
            x1, y1, x2, y2 = label[1:5]    
            true_cls = NAME_TO_ID.get(label[17], -1)

            if len(xyxy_np) > 0:
                ious = bbox_iou([x1, y1, x2, y2], xyxy_np)
                best_idx = np.argmax(ious)
                best_iou = ious[best_idx]

                if best_iou >= 0.5 and conf_np[best_idx] > 0.5:
                    pred_cls_name = model_26.names[cls_np[best_idx]]
                    pred_cls_id   = NAME_TO_ID.get(pred_cls_name, -1)
                    all_pred_cls.append(int(pred_cls_id))
                    all_gt_cls.append(int(true_cls))
                    print(f"GT: {CLASS_MAP[str(true_cls)]}, Pred: {pred_cls_name}, IoU: {best_iou:.2f}")
                else:
                    no_match_count += 1
                    print(f"GT: {CLASS_MAP[str(true_cls)]}, No matching YOLO box (IoU max: {best_iou:.2f})")
            else:
                no_match_count += 1
                print(f"GT: {CLASS_MAP[str(true_cls)]}, No YOLO boxes detected")
            
    print(f"Objets sans match YOLO (FN) : {no_match_count}")
    print(f"Objets évalués en classification : {len(all_gt_cls)}")
    print(f"Accuracy : {accuracy_score(all_gt_cls, all_pred_cls) * 100:.1f}%")
    print(classification_report(
        all_gt_cls, all_pred_cls,
        labels=[-1,0,1,2,3,4],
        target_names=['unknown']+CLASS_NAMES,
        zero_division=0
    ))
           


if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='FFTRadNet Evaluation')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    main(config)





    
