import os
import json
import argparse
import sys
from matplotlib import patches
from matplotlib.animation import FFMpegWriter
import torch
import random
import cv2
import numpy as np
from utils.util import process_predictions_FFT
from dataset.dataset import RADIal
from dataset.encoder import ra_encoder
from dataset.dataloader import CreateDataLoaders
import pkbar
import torch.nn.functional as F
from utils.evaluation import run_FullEvaluation
import torch.nn as nn
from model.RadViT import RadViT
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/skouff/RADIal/')
from DBReader.DBReader import SyncReader

N = 256
NR = 512
delta_v = 0.1
Vmax = delta_v * (N // 2)
delta_r = 0.201171875
velocity_bins = np.arange(N) * delta_v - Vmax
range_bins = np.arange(NR) * delta_r

CLASS_NAMES = ['car', 'truck', 'bicycle', 'bus', 'person']
BOX_W_DEG = 5.0   
BOX_H_M   = 5.0 

class RealTimeViewer:
    def __init__(self):
        self.paused = False

        self.fig, (self.ax_cam, self.ax_pred) = plt.subplots(1, 2, figsize=(18,5))

        self.im_cam = None
        self.im_pred = None
        self.title_pred = self.fig.suptitle("")
        self.cbar = None
        
        self.ax_cam.axis("off")

        # FIX AXES RA
        self.ax_pred.invert_xaxis()
        self.ax_pred.set_xlabel("Angle (°)")
        self.ax_pred.set_ylabel("Range (m)")

        # KEY PRESS
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        plt.ion()
        plt.tight_layout()
        plt.show()

    def on_key(self, event):
        if event.key == 'p':
            self.paused = not self.paused
            print("Paused" if self.paused else "Resuming")
        elif event.key == 'q':
            print("Quitting...")
            plt.close('all')
            os._exit(0)

    def update(self, image, conf_map, pred, labels, frame_idx):

        self.ax_pred.set_title(f"RA Prediction - Frame n°{frame_idx}")
        self.ax_cam.set_title(f"Camera - Frame n°{frame_idx}")
        
        if self.im_cam is None:
            self.im_cam = self.ax_cam.imshow(image)
        else: 
            self.im_cam.set_data(image)
        
        range_res_out = config['dataset']['geometry']['resolution'][0] * 4  # m/bin
        angle_res_out = config['dataset']['geometry']['resolution'][1] * 4  # °/bin

        range_max = conf_map.shape[0] * range_res_out
        angle_min = -conf_map.shape[1] / 2 * angle_res_out
        angle_max =  conf_map.shape[1] / 2 * angle_res_out

        if self.im_pred is None:
            self.im_pred = self.ax_pred.imshow(conf_map, aspect='auto', origin='lower', cmap='viridis',
                            vmin=0, vmax=1,
                            extent=[angle_min, angle_max, 0, range_max])
            self.ax_pred.set_xlim(angle_max, angle_min)
            self.cbar = self.fig.colorbar(self.im_pred, ax=self.ax_pred)
            self.cbar.set_label("Confidence")
        else:
            self.im_pred.set_data(conf_map)
        
        for p in self.ax_pred.patches:
            p.remove()
        for t in self.ax_pred.texts:
            t.remove()

        for label in labels:
            r_center = label[0]
            a_center = label[1]
            cls_id = int(label[10])
            rect = patches.Rectangle(
                (a_center - BOX_W_DEG / 2, r_center - BOX_H_M / 2),
                BOX_W_DEG, BOX_H_M,
                linewidth=1.5, edgecolor="green", facecolor='none', linestyle='-'
            )
            self.ax_pred.add_patch(rect)
            self.ax_pred.text(
                a_center, r_center + BOX_H_M / 2 + 0.3,
                CLASS_NAMES[cls_id],
                color="green", fontsize=7, va='bottom', ha='center'
            )

        for p in pred:
            r_center = p[9]
            a_center = p[10]
            cls_id = max(0, min(int(p[11]) if len(p) > 11 else 0, 4))
            rect = patches.Rectangle(
                (a_center - BOX_W_DEG / 2, r_center - BOX_H_M / 2),
                BOX_W_DEG, BOX_H_M,
                linewidth=1.5, edgecolor="red", facecolor='none', linestyle='--'
            )
            self.ax_pred.add_patch(rect)
            self.ax_pred.text(
                a_center, r_center + BOX_H_M / 2 + 0.3,
                CLASS_NAMES[cls_id],
                color="red", fontsize=7, va='bottom', ha='center'
            )
            
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        while self.paused:
            plt.pause(0.05)

def main(config, checkpoint,difficult):
    """
    Main function to run the evaluation inference visualization of the RadViT model on the RADIal dataset.
    Args: 
        config: Configuration dictionary loaded from a JSON file.
        checkpoint: Path to the model checkpoint to load.
        difficult: Whether to include difficult samples in the evaluation.
    """

    # Setup random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    # Load the dataset
    enc = ra_encoder(geometry = config['dataset']['geometry'],
                        statistics = config['dataset']['statistics'],
                        regression_layer = 2)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    parameters = config['model']['vit']
    net = nn.DataParallel(RadViT(parameters['D'], parameters['p'], parameters['H'], parameters['W'], parameters['neuron'], parameters['mha'], parameters['layer'], parameters['dropout'], parameters['n_encoders'], data_mode=config['data_mode']), device_ids=[0])
    net.to(device)
    dataset = RADIal(root_dir = config['dataset']['root_dir'],
                        statistics= config['dataset']['statistics'],
                        encoder=enc.encode,
                        difficult=difficult,perform_FFT=config['data_mode'])

    train_loader, val_loader, test_loader = CreateDataLoaders(dataset,config['dataloader'],config['seed'])
    dict = torch.load(checkpoint, weights_only=False)
    net.load_state_dict(dict['net_state_dict'])
    db_cache = {}
    
    viewer = RealTimeViewer()
    writer = FFMpegWriter(fps=2)

    output_video = f"/home/skouff/master_thesis/video.mp4"
    with writer.saving(viewer.fig, output_video, dpi=200):

        for i, data in enumerate(test_loader):
            
            labels_object = data[3][0].numpy()    

            if config['data_mode'] == 'ADC':
                inputs = data[0].to(device).type(torch.complex64)
            else:
                inputs = data[0].to(device).float()

            with torch.set_grad_enabled(False):
                outputs = net(inputs)

            sequence = data[5][0][0,:][0]
            frame_idx = data[5][0][0,:][1]
            if sequence not in db_cache:
                db_cache[sequence] = SyncReader(os.path.join("/Benson_DATA3/Public/RADIal/raw_sequences/", sequence), tolerance=20000, silent=True)
            
            data_reader = db_cache[sequence].GetSensorData(frame_idx)
            image = cv2.cvtColor(data_reader['camera']['data'], cv2.COLOR_BGR2RGB)
            
            out_obj = outputs['Detection'].detach().cpu().numpy().copy()
            pred = np.asarray(enc.decode(out_obj[0], 0.05))
            conf_map = out_obj[0][0]

            if(len(pred)>0):
                pred = process_predictions_FFT(pred, confidence_threshold=0.5)
            
            viewer.update(image, conf_map, pred, labels_object, frame_idx)
            writer.grab_frame()
            plt.pause(0.001)
    plt.ioff()
    plt.show()
    print("Video saved:", output_video)

        


if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='FFTRadNet Evaluation')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--checkpoint', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('--difficult', action='store_true')
    args = parser.parse_args()

    config = json.load(open(args.config))

    main(config, args.checkpoint,args.difficult)
