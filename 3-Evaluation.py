import os
import json
import argparse
import torch
import random
import numpy as np
from dataset.dataset import RADIal
from dataset.encoder import ra_encoder
from dataset.dataloader import CreateDataLoaders
import pkbar
import torch.nn.functional as F
from utils.evaluation import run_FullEvaluation
import torch.nn as nn
from model.RadViT import RadViT
def main(config, checkpoint,difficult):

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

    print("Parameters: ",sum(p.numel() for p in net.parameters() if p.requires_grad))

    print('===========  Loading the model ==================:')
    dict = torch.load(checkpoint, weights_only=False)
    net.load_state_dict(dict['net_state_dict'])

    print('===========  Running the evaluation ==================:')
    run_FullEvaluation(net,test_loader,enc,config=config, device=device)





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
