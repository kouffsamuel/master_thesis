import argparse
import json
import os
import random
import sys

import numpy as np
from shapely import Polygon
from sklearn.metrics import average_precision_score
import torch
from ultralytics import YOLO
import cv2

from dataset.dataset import RADIal
from dataset.dataloader import CreateDataLoaders
from dataset.encoder import ra_encoder

sys.path.insert(0, '/home/skouff/RADIal/')
from DBReader.DBReader import SyncReader

CLASS_TO_ID = {'car':0,'truck':1,'bicycle':2,'bus':3, 'person':4}
ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}
MIN_GT = 50

gt_count = {cls: 0 for cls in CLASS_TO_ID}

def bbox_iou(box1, boxes):
    x1, y1, x2, y2 = box1
    ious = []
    for box in boxes:
        x1b, y1b, x2b, y2b = box
        xi1 = max(x1, x1b); 
        yi1 = max(y1, y1b)
        xi2 = min(x2, x2b); 
        yi2 = min(y2, y2b)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box_area = (x2 - x1) * (y2 - y1)
        boxb_area = (x2b - x1b) * (y2b - y1b)
        union_area = box_area + boxb_area - inter_area
        iou = inter_area / union_area if union_area > 0 else 0
        ious.append(iou)

    return np.array(ious)

def compute_mAP(detections):
    APs = []
    for cls_id in CLASS_TO_ID.keys():

        if gt_count[cls_id] == 0:
            continue
        if gt_count[cls_id] < MIN_GT:
            continue

        cls_detections = [d for d in detections if d['pred_cls'] == CLASS_TO_ID[cls_id]]
        
        print(f"Class {cls_id}: {len(cls_detections)} detections")

        if len(cls_detections) == 0:
            APs.append(0.0)
            continue

        y_true = [1 if d['TP'] else 0 for d in cls_detections]
        y_scores = [d['conf'] for d in cls_detections]

        if(sum(y_true) == 0):
            APs.append(0.0)
            continue

        AP = average_precision_score(y_true, y_scores)
        APs.append(AP)
        print(f"AP {cls_id}: {AP:.4f}")
    return np.mean(APs)

def compute_detection_metrics(detections, gt_counts):
    TP = sum(1 for d in detections if d['TP'])
    FP = sum(1 for d in detections if not d['TP'])
    FN = sum(gt_counts.values()) - TP

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

def collect_detections(all_frames, iou_treshold=0.5):
    detections = []
    for xyxy_np, cls_ids, confs, gt_cls, gt_boxes in all_frames:
        used_gt = np.zeros(len(gt_boxes), dtype=bool)
        for i in range(len(xyxy_np)):
            pred_box = xyxy_np[i]
            iou = bbox_iou(pred_box, gt_boxes)
            ids = np.where(iou >= iou_treshold)[0]
            if len(ids) > 0:
                best_iou_idx = ids[np.argmax(iou[ids])]
                if used_gt[best_iou_idx] == 0:
                    used_gt[best_iou_idx] = 1
                    detections.append({
                        'conf': confs[i],
                        'pred_cls': cls_ids[i],
                        'gt_cls': int(gt_cls[best_iou_idx]),
                        'TP': (cls_ids[i] == int(gt_cls[best_iou_idx]))

                    })
                else:
                    detections.append({
                        'conf': confs[i],
                        'pred_cls': cls_ids[i],
                        'gt_cls': int(gt_cls[best_iou_idx]),
                        'TP': False
                    })
            else:
                detections.append({
                    'conf': confs[i],
                    'pred_cls': cls_ids[i],
                    'gt_cls': -1,
                    'TP': False
                })
 
    return detections

def main(config, difficult):

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    yolo26 = YOLO('/home/skouff/master_thesis/dataset/yolo26n.pt')

    enc = ra_encoder(
        geometry=config['dataset']['geometry'],
        statistics=config['dataset']['statistics'],
        regression_layer=2
    )

    dataset = RADIal(
        root_dir=config['dataset']['root_dir'],
        statistics=config['dataset']['statistics'],
        encoder=enc.encode,
        difficult=difficult,
        perform_FFT=config['data_mode']
    )

    train_loader, val_loader, test_loader = CreateDataLoaders(dataset, config['dataloader'], config['seed'])

    db_cache = {}
    all_frames = []
    for i, data in enumerate(test_loader):
        print(f"Processing sample {i+1}/{len(test_loader)}")

        labels = data[3]
        labels = labels[0].cpu().numpy()

        sequence = data[5][0][0,:][0]
        frame_idx = data[5][0][0,:][1]

        if sequence not in db_cache:
            db_cache[sequence] = SyncReader(os.path.join("/Benson_DATA3/Public/RADIal/raw_sequences/", sequence), tolerance=20000, silent=True)
        
        data_reader = db_cache[sequence].GetSensorData(frame_idx)
        image = data_reader['camera']['data']
        
        yolo26_results = yolo26(image, verbose=False, classes=[0, 1, 2, 5, 7])[0]
        cls_np  = yolo26_results.boxes.cls.cpu().numpy().astype(int)
        xyxy_np = yolo26_results.boxes.xyxy.cpu().numpy()
        conf_np = yolo26_results.boxes.conf.cpu().numpy()
        

        all_frames.append((xyxy_np, 
                           [CLASS_TO_ID.get(yolo26_results.names[cls], -1) for cls in cls_np], 
                           conf_np, 
                           labels[:, 10],
                           labels[:, 6:10]))
    
    for *_, gt_cls, _ in all_frames:
        for cls_id in gt_cls.astype(int):
            name = ID_TO_CLASS.get(cls_id)
            if name:
                gt_count[name] += 1   
    
    print("Ground Truth Counts per Class:")
    for cls, count in gt_count.items():
        print(f"{cls}: {count}")

    detections = collect_detections(all_frames, iou_treshold=0.5)
    # p, r, f1 = compute_detection_metrics(detections, gt_count)
    # print(f"Yolo Detection  - Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")

    thresholds = np.arange(0.5, 1.0, 0.05)
    print(f"RadViT Evaluation :")
    print("======== AP@50: ========")
    mAP50 = compute_mAP(detections)
    print(f"RadViT mAP@50: {mAP50:.4f}")
    mAP50_95 = []
    for thresh in thresholds:
        print(f"======== AP@{thresh*100:.0f}: ========")
        mAP = compute_mAP(collect_detections(all_frames, thresh))
        mAP50_95.append(mAP)

    print(f"RadViT mAP@50:95: {np.mean(mAP50_95):.4f}")
    print()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FFTRadNet Evaluation')
    parser.add_argument('-c', '--config', default='config.json', type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('--difficult', action='store_true')
    args = parser.parse_args()

    config = json.load(open(args.config))
    main(config, args.difficult)