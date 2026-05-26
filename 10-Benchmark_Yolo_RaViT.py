import argparse
import json
import os
import pickle
import random
import sys

from matplotlib import pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
import torch
from ultralytics import YOLO
from utils.util import imageToWorld, worldToImage
from utils.metrics import RA_to_cartesian_box, bbox_iou, process_predictions_FFT
import cv2
from dataset.encoder import ra_encoder
from model.RadViT import RadViT
import torch.nn as nn
from dataset.dataset import RADIal
from dataset.dataloader import CreateDataLoaders
from sklearn.metrics import average_precision_score, precision_recall_curve, confusion_matrix

sys.path.insert(0, '/home/skouff/RADIal/')
from DBReader.DBReader import SyncReader

CLASS_TO_ID = {'car':0,'truck':1,'bicycle':2,'bus':3, 'person':4}
ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}
MIN_GT = 50

gt_count = {'car':877,'truck':376,'bicycle':6,'bus':0, 'person':1}


def yolo_bbox_to_RA(xyxy):
    u_c = (xyxy[0] + xyxy[2]) / 2
    v_c = xyxy[3]

    x, y, _ = imageToWorld(u_c, v_c, z_world=0.0)  # hauteur du radar

    range_m = np.sqrt(x**2 + y**2)
    angle   = np.degrees(np.arctan2(-x, y))

    return range_m, angle


def collect_detections(all_frames, iou_treshold=0.5):
    radar_detections = []
    yolo_detections = []

    for (object_pred, true_obj, ground_truth_box_corners, 
         yolo_box_corners, ground_truth_box_gt_corners, yolo_cls, yolo_conf) in all_frames:
            
            if len(ground_truth_box_gt_corners) > 0 and len(yolo_box_corners) > 0:
                used_gt_yolo = np.zeros(len(ground_truth_box_gt_corners))

                for yolo_box_id, prediction_yolo in enumerate(yolo_box_corners):
                    iou = bbox_iou(prediction_yolo, ground_truth_box_gt_corners)
                    ids = np.where(iou >= iou_treshold)[0]
                    if len(ids) > 0:
                        best_pred = ids[np.argmax(iou[ids])]
                        if used_gt_yolo[best_pred] == 0:
                            used_gt_yolo[best_pred] = 1
                            yolo_detections.append({
                                'conf': yolo_conf[yolo_box_id],
                                'pred_cls': yolo_cls[yolo_box_id],
                                'gt_cls': int(true_obj[best_pred, -1]),
                                'TP': yolo_cls[yolo_box_id] == int(true_obj[best_pred, -1])
                            })
                        else:
                            yolo_detections.append({
                                'conf': yolo_conf[yolo_box_id],
                                'pred_cls': yolo_cls[yolo_box_id],
                                'gt_cls': int(true_obj[best_pred, -1]),
                                'TP': False
                            })
                    else:
                        yolo_detections.append({
                            'conf': yolo_conf[yolo_box_id],
                            'pred_cls': yolo_cls[yolo_box_id],
                            'gt_cls': -1,
                            'TP': False
                        })
                
            
            if len(ground_truth_box_corners) > 0 and len(object_pred) > 0:
                used_gt = np.zeros(len(ground_truth_box_corners))
                for prediction in object_pred:
                    iou = bbox_iou(prediction[1:9], ground_truth_box_corners)
                    ids = np.where(iou >= iou_treshold)[0]
                    if len(ids) > 0:
                        best_pred = ids[np.argmax(iou[ids])]
                        if used_gt[best_pred] == 0:
                            used_gt[best_pred] = 1
                            radar_detections.append({
                                'conf': float(prediction[0]),
                                'pred_cls': int(prediction[-1]),
                                'gt_cls': int(true_obj[best_pred, -1]),
                                'det_TP': True,
                                'cls_TP': (int(prediction[-1]) == int(true_obj[best_pred, -1]))
                            })
                        else: 
                            radar_detections.append({
                                'conf': float(prediction[0]),
                                'pred_cls': int(prediction[-1]),
                                'gt_cls': int(true_obj[best_pred, -1]),
                                'det_TP': False,
                                'cls_TP': False
                            })
                    else:
                       radar_detections.append({
                            'conf': float(prediction[0]),
                            'pred_cls': int(prediction[-1]),
                            'gt_cls': -1,
                            'det_TP': False,
                            'cls_TP': False
                        })
    return radar_detections, yolo_detections    

def compute_mAP(detections):
    APs = []
    for cls_id in CLASS_TO_ID.keys():
        if gt_count[cls_id] == 0:
            continue

        if gt_count[cls_id] < MIN_GT:
            continue

        cls_detections = [d for d in detections if d['pred_cls'] == CLASS_TO_ID[cls_id]]
        # print(f"Class {cls_id}: {len(cls_detections)} detections")

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

def compute_confusion_matrix(detections, gt_counts):
    y_true = []
    y_pred = []
    
    for d in detections:
        if d['gt_cls'] != -1 and d['TP'] == True: 
            if gt_counts[ID_TO_CLASS[d['gt_cls']]] < MIN_GT:
                continue
            if gt_counts[ID_TO_CLASS[d['gt_cls']]] == 0:
                continue

            y_true.append(d['gt_cls'])
            y_pred.append(d['pred_cls'])

    return confusion_matrix(y_true, y_pred)


def main(config, checkpoint, difficult):

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    enc = ra_encoder(
        geometry=config['dataset']['geometry'],
        statistics=config['dataset']['statistics'],
        regression_layer=2
    )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    parameters = config['model']['vit']
    
    net = RadViT(
        parameters['D'], parameters['p'], parameters['H'], parameters['W'],
        parameters['neuron'], parameters['mha'], parameters['layer'],
        parameters['dropout'], parameters['n_encoders'],
        data_mode=config['data_mode']
    )

    net.to(device)

    yolo26 = YOLO('/home/skouff/master_thesis/dataset/yolo26x.pt')

    db_cache = {}


    checkpoint_data = torch.load(checkpoint, weights_only=False)
    model_state_dict = {k.replace('module.', ''): v for k, v in checkpoint_data['net_state_dict'].items()}
    net.load_state_dict(model_state_dict)

    dataset = RADIal(
        root_dir=config['dataset']['root_dir'],
        statistics=config['dataset']['statistics'],
        encoder=enc.encode,
        difficult=difficult,
        perform_FFT=config['data_mode']
    )

    train_loader, val_loader, test_loader = CreateDataLoaders(dataset, config['dataloader'], config['seed'])

    net.eval()

    all_frames = []

    for i, data in enumerate(test_loader):
        print(f"Processing sample {i+1}/{len(test_loader)}")
        
        if config['data_mode'] == 'ADC':
            inputs = data[0].to(device).type(torch.complex64)
        else:
            inputs = data[0].to(device).float()

        labels = data[3]

        sequence = data[5][0][0,:][0]
        frame_idx = data[5][0][0,:][1]
        if sequence not in db_cache:
            db_cache[sequence] = SyncReader(os.path.join("/Benson_DATA3/Public/RADIal/raw_sequences/", sequence), tolerance=20000, silent=True)
        
        data_reader = db_cache[sequence].GetSensorData(frame_idx)
        image = data_reader['camera']['data']

        with torch.set_grad_enabled(False):
            outputs = net(inputs)
        
        yolo26_results = yolo26(image, verbose=False, classes=[0, 1, 2, 5, 7])[0]

        cls_np  = yolo26_results.boxes.cls.cpu().numpy().astype(int)
        xyxy_np = yolo26_results.boxes.xyxy.cpu().numpy()
        conf_np = yolo26_results.boxes.conf.cpu().numpy()

        out_obj = outputs['Detection'].detach().cpu().numpy().copy()
        
        yolo_box_corners = []
        yolo_cls = []
        yolo_conf = []

        # Convert YOLO detections to Cartesian coordinates and filter them based on range and angle
        for xyxy, cls_id, conf in zip(xyxy_np, cls_np, conf_np):
            range, angle = yolo_bbox_to_RA(xyxy)
            if not(-90 <= angle <= 90) or not (range >= 5 and range <= 100):
                continue
            bboxe = np.asarray(RA_to_cartesian_box([[range, angle]]))
            yolo_box_corners.append(bboxe)
            yolo_cls.append(CLASS_TO_ID.get(yolo26_results.names[cls_id], -1))  # Map class name to ID, default to -1 if not found
            yolo_conf.append(conf)
            
        yolo_box_corners = np.asarray(yolo_box_corners)
    
        # Process radar predictions and ground truth, and collect detections for evaluation
        for pred_obj,true_obj in zip(out_obj,labels):

            object_pred = []
            ground_truth_box_corners = []
            ground_truth_box_gt_corners = []
            
            pred_decoded = np.asarray(enc.decode(pred_obj,0.05))

            if len(pred_decoded) > 0:
                object_pred = process_predictions_FFT(pred_decoded, confidence_threshold=0.5)
            
            if len(object_pred) > 0:
                dist = (object_pred[:, 2] + object_pred[:, 4]) / 2
                ids  = np.where((dist >= 5) & (dist <= 100))
                object_pred = object_pred[ids]
            
            if len(true_obj) > 0:
                ids = np.where((true_obj[:, 0] >= 5) & (true_obj[:, 0] <= 100))
                true_obj = true_obj[ids]
                ground_truth_box_corners = np.asarray(RA_to_cartesian_box(true_obj))
                
                # Convert GT bboxes from camera to radar coordinates for IoU calculation with YOLO predictions
                for obj in true_obj:
                    # gt_count[ID_TO_CLASS.get(int(obj[-1]))] += 1
                    x1, y1, x2, y2 = obj[6:10]
                    u_c = (x1 + x2) / 2
                    v_c = y2
                    x, y, _ = imageToWorld(u_c, v_c, z_world=0.0)
                    range_m = np.sqrt(x**2 + y**2)
                    angle   = np.degrees(np.arctan2(-x, y))
                    boxe = RA_to_cartesian_box([[range_m, angle]])
                    
                    ground_truth_box_gt_corners.append(boxe)

            ground_truth_box_gt_corners = np.asarray(ground_truth_box_gt_corners)
            all_frames.append((object_pred, true_obj, ground_truth_box_corners, yolo_box_corners, ground_truth_box_gt_corners, yolo_cls, yolo_conf))
        
        # with open("all_frames_yolo26n.pkl", "wb") as f:
        #     pickle.dump(all_frames, f)

    radar_detections, yolo_detections = collect_detections(all_frames, iou_treshold=0.5)

    p, r, f1 = compute_detection_metrics(radar_detections, gt_count)
    print(f"RadViT Detection  - Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")

    p, r, f1 = compute_detection_metrics(yolo_detections, gt_count)
    print(f"YOLO   Detection  - Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")

    # thresholds = np.arange(0.5, 1.0, 0.05)
    # print(f"RadViT Evaluation :")
    # print("======== AP@50: ========")
    # mAP50 = compute_mAP(radar_detections)
    # print(f"RadViT mAP@50: {mAP50:.4f}")
    # mAP50_95 = []
    # for thresh in thresholds:
    #     print(f"======== AP@{thresh*100:.0f}: ========")
    #     mAP = compute_mAP(collect_detections(all_frames, thresh)[0])
    #     mAP50_95.append(mAP)

    # print(f"RadViT mAP@50:95: {np.mean(mAP50_95):.4f}")
    # print()

    # print(f"Yolo26x Evaluation :")
    # print("======== AP@50: ========")
    # mAP50 = compute_mAP(yolo_detections)
    # print(f"Yolo26x mAP@50: {mAP50:.4f}")
    # mAP50_95 = []
    # for thresh in thresholds:
    #     print(f"======== AP@{thresh*100:.0f}: ========")
    #     mAP = compute_mAP(collect_detections(all_frames, thresh)[1])
    #     mAP50_95.append(mAP)

    # print(f"Yolo26x mAP@50:95: {np.mean(mAP50_95):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FFTRadNet Evaluation')
    parser.add_argument('-c', '--config', default='config.json', type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--checkpoint', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('--difficult', action='store_true')
    args = parser.parse_args()

    config = json.load(open(args.config))
    main(config, args.checkpoint, args.difficult)