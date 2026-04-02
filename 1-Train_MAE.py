import glob
import os
import json
import argparse
import re
import torch
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from utils import lr_sched
from model.MAE_RadViT import MAE_RadViT
from model.MVIT import MViT
from model.RadViT import RadViT
#from model.RadViT_newconfig import RadViT
from model.MVIT_ADC import MViT_ADC
from dataset.dataset import RADIal
from dataset.encoder import ra_encoder
from dataset.dataloader import CreateDataLoaders
import pkbar
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from loss import pixor_loss
from utils.evaluation import run_evaluation
import torch.nn as nn

def add_weight_decay(model, weight_decay):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0}
    ]


def main(config=None, resume=None, exp_name=None): 
    # Setup random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    output_folder = config['output']['dir']#Path(config['output']['dir'])

    # Create directory structure
    if not os.path.exists(os.path.join(output_folder,exp_name)):
        os.mkdir(os.path.join(output_folder,exp_name))

    with open(os.path.join(output_folder,exp_name,'config.json'),'w') as outfile:
        json.dump(config, outfile)

    writer = SummaryWriter(os.path.join(output_folder, exp_name))

    enc = ra_encoder(geometry = config['dataset']['geometry'],
                        statistics = config['dataset']['statistics'],
                        regression_layer = 2)


    # Create the model
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    
    mae_radvit = MAE_RadViT(norm_pix_loss=True)
    
    mae_radvit.to(device)

    dataset = RADIal(root_dir = config['dataset']['root_dir'],
                        statistics= config['dataset']['statistics'],
                        encoder=enc.encode,
                        difficult=True,perform_FFT=config['data_mode'])


    train_loader, val_loader, test_loader = CreateDataLoaders(dataset,config['dataloader'],config['seed'])

    # Optimizer
    lr = float(config['optimizer']['lr'])
    step_size = int(config['lr_scheduler']['step_size'])
    gamma = float(config['lr_scheduler']['gamma'])
    num_epochs=int(config['num_epochs'])
    warmup_epochs = 10
    mask_ratio = 0.75

    param_groups = add_weight_decay(mae_radvit, 0.05)
    optimizer = optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95)) 
    scheduler_warmup = lr_scheduler.LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs)
    scheduler_cosine = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs, eta_min=lr * 1e-2)
    scheduler = lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_epochs])

    print('===========  Optimizer  ==================:')
    print('      LR:', lr)
    print('      step_size:', step_size)
    print('      gamma:', gamma)
    print('      num_epochs:', num_epochs)
    print('      warmup_epochs:', warmup_epochs)
    print('')

    # Train
    startEpoch = 0
    global_step = 0
    history = {'train_loss':[],'val_loss':[],'lr':[],'mAP':[],'mAR':[]}


    if resume:
        print('===========  Resume training  ==================:')
        dict = torch.load(resume, weights_only=False)
        mae_radvit.load_state_dict(dict['net_state_dict'])
        optimizer.load_state_dict(dict['optimizer'])
        startEpoch = dict['epoch']+1
        history = dict['history']
        global_step = dict['global_step']

        print('       ... Start at epoch:',startEpoch)


    for epoch in range(startEpoch,num_epochs):

        kbar = pkbar.Kbar(target=len(train_loader), epoch=epoch, num_epochs=num_epochs, width=20, always_stateful=False)

        ###################
        ## Training loop ##
        ###################
        mae_radvit.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            if config['data_mode'] == 'ADC':
                inputs = data[0].to(device).type(torch.complex64)
            else:
                inputs = data[0].to(device).float()
                
            # reset the gradient
            optimizer.zero_grad()

            # forward pass, enable to track our gradient
            with torch.set_grad_enabled(True):
                loss, _, _ = mae_radvit(imgs=inputs, mask_ratio=mask_ratio)

            loss_value = loss.item()

            writer.add_scalar('Loss/train', loss_value, global_step)

            # backprop
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss_value

            kbar.update(i, values=[("loss",loss_value)])

            global_step += 1


        history['train_loss'].append(running_loss / len(train_loader))
        mae_radvit.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs = data[0].to(device).float()
                loss, _, _ = mae_radvit(imgs=inputs, mask_ratio=mask_ratio)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        history['val_loss'].append(val_loss)
        writer.add_scalar('Loss/val', val_loss, global_step)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
        
        scheduler.step()
        history['lr'].append(scheduler.get_last_lr()[0])
        
        checkpoint = {}
        checkpoint['net_state_dict'] = mae_radvit.state_dict()
        checkpoint['encoder_state_dict'] = mae_radvit.encoder_layers.state_dict()
        checkpoint['optimizer'] = optimizer.state_dict()
        checkpoint['scheduler'] = scheduler.state_dict()
        checkpoint['epoch'] = epoch
        checkpoint['history'] = history
        checkpoint['global_step'] = global_step

        filename = os.path.join(output_folder, exp_name, 'checkpoint_epoch_{:04d}.pth'.format(epoch))
        torch.save(checkpoint,filename)

        print('')


if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='FFTRadNet Training')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-n', '--name', default='exp', type=str,
                        help='Name of the experiment (default: exp)')

    args = parser.parse_args()

    config = json.load(open(args.config))
    main(config, args.resume, args.name)

