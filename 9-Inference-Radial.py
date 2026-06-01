import os
import json
import argparse
import torch
import random
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

N = 256
NR = 512
delta_v = 0.1
Vmax = delta_v * (N // 2)
delta_r = 0.201171875
velocity_bins = np.arange(N) * delta_v - Vmax
range_bins = np.arange(N) * delta_r

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
    
    for i, data in enumerate(test_loader):

        if config['data_mode'] == 'ADC':
            inputs = data[0].to(device).type(torch.complex64)
        else:
            inputs = data[0].to(device).float()

        with torch.set_grad_enabled(False):
            outputs = net(inputs)
        
        CLASS_COLORS = ['red', 'orange', 'purple', 'cyan', 'magenta']
        CLASS_NAMES = ['car', 'truck', 'bicycle', 'bus', 'person']

        out_obj = outputs['Detection'].detach().cpu().numpy().copy()

        labels_object = data[3]    
        camera_image = data[4][0].numpy()  
        conf_map = out_obj[0][0]

        current_pred = np.asarray(enc.decode(out_obj[0], 0.5))
        current_true = labels_object[0].numpy()

        fig, axes = plt.subplots(1, 2, figsize=(18, 5))

        axes[0].imshow(camera_image)
        axes[0].set_title("Camera Input")
        axes[0].axis('off')

        range_res_out = config['dataset']['geometry']['resolution'][0] * 4  # m/bin
        angle_res_out = config['dataset']['geometry']['resolution'][1] * 4  # °/bin

        range_max = conf_map.shape[0] * range_res_out
        angle_min = -conf_map.shape[1] / 2 * angle_res_out
        angle_max =  conf_map.shape[1] / 2 * angle_res_out

        im = axes[1].imshow(conf_map, aspect='auto', origin='lower', cmap='viridis',
                            vmin=0, vmax=1,
                            extent=[angle_min, angle_max, 0, range_max])

        true_classes = set()
        for j in range(len(current_true)):
            cls_id = max(0, min(int(current_true[j][10]), len(CLASS_NAMES) - 1))
            label = f'True: {CLASS_NAMES[cls_id]}' if cls_id not in true_classes else ""
            true_classes.add(cls_id)
            axes[1].scatter(current_true[j][1], current_true[j][0],  # angle (°), range (m)
                            c='green', marker='o', s=80, label=label)

        if len(current_pred) > 0:
            predicted_classes = set()
            for pred in current_pred:
                cls_id = max(0, min(int(pred[5]) if len(pred) > 5 else 0, len(CLASS_NAMES) - 1))
                color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
                label = f'Pred: {CLASS_NAMES[cls_id]}' if cls_id not in predicted_classes else ""
                predicted_classes.add(cls_id)
                axes[1].scatter(pred[1], pred[0],  c=color, marker='x', s=80, alpha=0.6, label=label)

        axes[1].set_title("Predicted Range-Azimuth Map")
        axes[1].set_xlabel("Angle (°)")
        axes[1].set_ylabel("Range (m)")
        plt.colorbar(im, ax=axes[1], label='Confidence')
        axes[1].legend(loc='upper right', fontsize=7)

        plt.tight_layout()
        plt.savefig(f"output.png")
        plt.close()



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
