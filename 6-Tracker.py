import os
import cv2
import json
import argparse
import sys
from matplotlib import gridspec, patches
import pandas as pd
import torch
import random
import numpy as np
from utils.metrics import process_predictions_FFT, RA_to_cartesian_box
from dataset.dataset import RADIal
# from dataset.encoder import ra_encoder
from torch.utils.data import Dataset, DataLoader, DistributedSampler, random_split,Subset
import torch.nn.functional as F
from utils.evaluation import run_FullEvaluation
import torch.nn as nn
from model.RadViT import RadViT
from dataset.dataloader import RADIal_collate
sys.path.append('/home/skouff/RADIal')
sys.path.append('/home/skouff/T_FFTRadNet')
from DBReader.DBReader import SyncReader
from SignalProcessing import RadarSignalProcessing
from RadIal.model.FFTRadNet_ViT import FFTRadNet_ViT
from RadIal.dataset.encoder import ra_encoder
import matplotlib.pyplot as plt
from utils.tracking import MOTEvaluator, MultiObjectTracker, Tracker
from utils.sort import Sort

ID_TO_CLASS = {0:'car',1:'truck',2:'bicycle',3:'bus', 4:'person'}

SEQUENCES = ['RECORD@2020-11-22_12.45.05',
             'RECORD@2020-11-22_12.25.47',
             'RECORD@2020-11-22_12.03.47',
             'RECORD@2020-11-22_12.54.38']


class RealTimeViewer:
    def __init__(self):
        self.paused = False
        self.fig, (self.ax_pred, self.ax_cam) = plt.subplots(1, 2, figsize=(12, 6))

        self.im_pred = None
        self.im_cam = None
        self.cbar = None
        self.title_pred = self.ax_pred.set_title("")

        # FIX AXES
        self.ax_pred.set_xlim(-50, 50)
        self.ax_pred.set_ylim(0, 100)
        self.ax_pred.set_xlabel("x (m)")
        self.ax_pred.set_ylabel("y (m)")
        self.ax_pred.invert_xaxis()
        self.ax_pred.grid(True)
        self.cmap = plt.get_cmap("tab20")

        self.ax_cam.axis("off")

        # KEY PRESS
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        plt.ion()
        plt.show()

    def on_key(self, event):
        if event.key == 'p':
            self.paused = not self.paused
            print("Paused" if self.paused else "Resuming")
        elif event.key == 'q':
            print("Quitting...")
            plt.close('all')
            os._exit(0)

    def update(self, sort, kalman, pred_boxes, img, tracks=None, frame_idx=0):

        self.ax_pred.cla()
        self.ax_pred.set_xlim(-50, 50)
        self.ax_pred.set_ylim(0, 100)
        self.ax_pred.set_xlabel("x (m)")
        self.ax_pred.set_ylabel("y (m)")
        self.ax_pred.invert_xaxis()
        self.ax_pred.grid(True)
        
        if len(pred_boxes) > 0 and kalman:
            boxes = pred_boxes[:, 1:9]
            # labels = pred_boxes[:, -1]
        elif sort:
            boxes = pred_boxes
        else:  
            boxes = []

        for i, box in enumerate(boxes):
            if kalman:
                corners = np.array(box).reshape(4, 2)
                polygon = plt.Polygon(
                    corners, closed=True,
                    edgecolor='red', facecolor='none', linewidth=1.5
                )
                self.ax_pred.add_patch(polygon)
            else:
                rect = patches.Rectangle(
                    (box[0], box[1]), box[2]-box[0], box[3]-box[1],
                    edgecolor="red", facecolor='none', linewidth=1.5
                )
                self.ax_pred.add_patch(rect)
            # cx, cy = box.mean(axis=0)
            # self.ax_pred.text(
            #     cx, cy + 1.5,          
            #     ID_TO_CLASS.get(int(labels[i])),
            #     color='red',
            #     fontsize=8,
            #     ha='center', va='bottom'
            # )
        
        if tracks is not None:
            for obj in tracks:
                cx, cy = obj['centroid']
                self.ax_pred.plot(cx, cy, "go", color=self.cmap(obj['id'] % self.cmap.N), markersize=2, label=f"ID{obj['id']}")
                self.ax_pred.legend(loc='upper right', markerscale=0.5)
        # ================= CAMERA =================
        if self.im_cam is None:
            self.im_cam = self.ax_cam.imshow(img)
        else:
            self.im_cam.set_data(img)

        self.ax_cam.set_title(f"Camera Frame {frame_idx}")
        self.ax_pred.set_title(f"Predicted Boxes - Frame {frame_idx}")

        # refresh rapide
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        # pause si demandé
        while self.paused:
            plt.pause(0.05)

def corners_to_bbox(corners):
    # Claude.ai generated function 
    # corners: array de shape (8,) ou (4,2)
    score = corners[0]
    coords = corners[1:9].reshape(4, 2)
    x1, y1 = coords[:, 0].min(), coords[:, 1].min()
    x2, y2 = coords[:, 0].max(), coords[:, 1].max()
    return np.array([x1, y1, x2, y2, score])

def main(config, checkpoint, sort=False, kalman=False):
    """
    Main function to run the tracking evaluation on the RADIal dataset.
    Args:
        config: Configuration dictionary loaded from a JSON file.
        checkpoint: Path to the model checkpoint to load.
    """

    # Setup random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # parameters = config['model']['vit']
    # net = RadViT(parameters['D'], parameters['p'], parameters['H'], parameters['W'], parameters['neuron'], 
    #              parameters['mha'], parameters['layer'], parameters['dropout'], parameters['n_encoders'], 
    #              data_mode=config['data_mode'])
    net = FFTRadNet_ViT(patch_size = config['model']['patch_size'],
                        channels = config['model']['channels'],
                        in_chans = config['model']['in_chans'],
                        embed_dim = config['model']['embed_dim'],
                        depths = config['model']['depths'],
                        num_heads = config['model']['num_heads'],
                        drop_rates = config['model']['drop_rates'],
                        regression_layer = 2,
                        detection_head = config['model']['DetectionHead'],
                        segmentation_head = config['model']['SegmentationHead'])

    net.to(device)

    enc = ra_encoder(
        geometry=config['dataset']['geometry'],
        statistics=config['dataset']['statistics'],
        regression_layer=2
    )

    dict = torch.load(checkpoint, weights_only=False)
    # dict = {k.replace('module.', ''): v for k, v in dict['net_state_dict'].items()}
    net.load_state_dict(dict['net_state_dict'])
    net.eval()

    viewer = RealTimeViewer()

    gt_tracks = pd.read_csv("/home/skouff/master_thesis/gt_tracks_manual.csv")
    all_accs = []
    for i, sequence in enumerate(SEQUENCES):
        evaluator = MOTEvaluator()
        print(f"Processing sequence {sequence} ({i+1}/{len(SEQUENCES)})")

        if sort:
            tracker = Sort(max_age=8, min_hits=4, iou_threshold=0.3)
        if kalman:
            tracker = MultiObjectTracker(max_misses=8, min_hits=4, mahal_threshold=9.21)

        Tracker._id_counter = 0 
        db = SyncReader( os.path.join('/Benson_DATA3/Public/RADIal/raw_sequences/', sequence), tolerance=20000, silent=True)
        RSP = RadarSignalProcessing('/home/skouff/master_thesis/CalibrationTable.npy', method='RD', shift=True, device=device)
        ite = iter(db)

        for j in range(len(db)):
            try:
                data = next(ite)
            except IndexError:
                print(f"End of sequence {sequence}")
                break

            img = data['camera']['data']
            idx = data['radar_ch0']['index']
            gt_frame = gt_tracks[(gt_tracks['index'] == idx) & (gt_tracks['dataset'] == sequence)]
            rd = RSP.run(data['radar_ch0']['data'],data['radar_ch1']['data'],data['radar_ch2']['data'],data['radar_ch3']['data'])
            
            for k in range(len(config["dataset"]["statistics"]['input_mean'])):
                rd[...,k] -= config["dataset"]["statistics"]['input_mean'][k]
                rd[...,k] /= config["dataset"]["statistics"]['input_std'][k]
            
            with torch.no_grad():
                rd_tensor = torch.from_numpy(rd).permute(2,0,1).unsqueeze(0).to(device).float()
                outputs = net(rd_tensor) 
               
            out_obj = outputs['Detection'].detach().cpu().numpy().copy()

            predictions = []
            for pred_obj in out_obj:
                pred_decoded = np.asarray(enc.decode(pred_obj,0.05))

                if len(pred_decoded) > 0:
                    object_pred = process_predictions_FFT(pred_decoded, confidence_threshold=0.5)
                else:
                    object_pred = []
                
                if len(object_pred) > 0:
                    dist = (object_pred[:, 2] + object_pred[:, 4]) / 2
                    ids  = np.where((dist >= 5) & (dist <= 100))
                    object_pred = object_pred[ids]
                
                predictions.append(object_pred)

            predictions = np.concatenate(predictions, axis=0)
            if sort: 
                # For Sort tracker, convert boxes to format [x1, y1, x2, y2, conf]
                predictions = np.array([corners_to_bbox(pred[:-1]) for pred in predictions])
                if predictions.shape[0] == 0:
                    predictions = np.empty((0, 5))

            active = tracker.update(predictions)
            
            if sort:
                active[:,4] -=1
                active_tracks = [{ 'centroid': [(t[0] + t[2]) / 2, (t[1] + t[3]) / 2], 'id': int(t[4])} for t in active]
                active = active_tracks

            viewer.update(sort, kalman, pred_boxes=predictions, img=img, tracks=active, frame_idx=j)
            if not gt_frame.empty:
                evaluator.update(gt_frame, active, frame_idx=idx)
            
            # for obj in active:
            #     print(f"Frame {j} | ID {obj['id']:3d} | "
            #         f"centroid ({obj['centroid'][0]:.1f}, {obj['centroid'][1]:.1f}) m | "
            #         f"hits={obj['hits']}")
        all_accs.append(evaluator.acc)
    # Final evaluation
    evaluator.summary(all_accs, SEQUENCES)
            

if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='FFTRadNet Evaluation')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--checkpoint', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('--sort', action='store_true', help='Use SORT tracker instead of Kalman', default=False)
    parser.add_argument('--kalman', action='store_true', help='Use Kalman tracker', default=False)
    args = parser.parse_args()

    config = json.load(open(args.config))

    main(config, args.checkpoint, sort=args.sort, kalman=args.kalman)
