import os
import json
import argparse
import torch
import numpy as np
from model.RadViT import RadViT
from dataset.dataset import RADIal
from dataset.encoder import ra_encoder
import cv2
from utils.util import DisplayHMI
import torch.nn as nn

def main(config, checkpoint_filename,difficult):

    # set device

    # Load the dataset
    enc = ra_encoder(geometry = config['dataset']['geometry'],
                        statistics = config['dataset']['statistics'],
                        regression_layer = 2)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    parameters = config['model']['vit']
    net = nn.DataParallel(RadViT(parameters['D'], parameters['p'], parameters['H'], parameters['W'], parameters['neuron'], parameters['mha'], parameters['layer'], parameters['dropout'], parameters['n_encoders']), device_ids=[0,1,2,3]) 
    net.to(device)
    dataset = RADIal(root_dir = config['dataset']['root_dir'],
                        statistics= config['dataset']['statistics'],
                        encoder=enc.encode,
                        difficult=difficult,perform_FFT=config['data_mode'])
    # Load the model
    dict = torch.load(checkpoint_filename, weights_only=False)
    net.load_state_dict(dict['net_state_dict'])
    net.eval()


    for data in dataset:
        # Display GD and predictions on HMI
        # Display bounding boxes on radar and camera image 

        # data is composed of [radar_FFT, segmap,out_label,box_labels,image]
        inputs = torch.tensor(data[0]).permute(2,0,1).to(device).unsqueeze(0)
        with torch.set_grad_enabled(False):
            outputs = net(inputs)
            if config['data_mode'] == 'ADC':
                intermediate = net.DFT(inputs).detach().cpu().numpy()[0]
            else:
                intermediate = None

        hmi = DisplayHMI(data[4], data[0], data[3], outputs,enc,config,intermediate)

        cv2.imshow('FFTRadNet',hmi)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='FFTRadNet test')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--checkpoint', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('--difficult', action='store_true')
    args = parser.parse_args()

    config = json.load(open(args.config))

    main(config, args.checkpoint,args.difficult)
