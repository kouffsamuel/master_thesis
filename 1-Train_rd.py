import os
import json
import argparse
import torch
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from model.MVIT import MViT
from dataset.dataset import RADIal
from dataset.encoder import rd_encoder
from dataset.dataloader import CreateDataLoaders
import pkbar
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from loss import pixor_loss
from utils.evaluation import run_evaluation
import torch.nn as nn

def main(config=None, resume=None, exp_name=None): 
    # Setup random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    output_folder = config['output']['dir']#Path(config['output']['dir'])

    # Create directory structure
    if not os.path.exists(os.path.join(output_folder,exp_name)):
        os.mkdir(os.path.join(output_folder,exp_name))

    with open(os.path.join(output_folder,exp_name,'config.json'),'w') as outfile:
        json.dump(config, outfile)

    writer = SummaryWriter(os.path.join(output_folder, exp_name))

    enc = rd_encoder(geometry = config['dataset']['geometry'],
                        statistics = config['dataset']['statistics'],
                        regression_layer = 2)


    # Create the model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    parameters = config['model']['vit']
    net = nn.DataParallel(MViT(parameters['D'], parameters['p'], parameters['H'], parameters['W'], parameters['neuron'], parameters['mha'], parameters['layer'], parameters['dropout']), device_ids=[0,1,2,3]) 
    net.to(device)
    # checkpoint = torch.load("/home/skouff/master_thesis/model/RADIal_SwinTransformer_RD_Shift.pth", weights_only=False, map_location='cpu')
    # model_state_dict = {k: v for k, v in checkpoint['net_state_dict'].items() if k.startswith('detection')}
    # net.load_state_dict(model_state_dict, strict=False)

    # Freeze RA Decoder and Detection Head at initialization
    # for name, param in net.named_parameters():
    #     if name.startswith('module.RA') or name.startswith('module.detection'):
    #         print(f"Loading {name}")

    dataset = RADIal(root_dir = config['dataset']['root_dir'],
                        statistics= config['dataset']['statistics'],
                        encoder=enc.encode,
                        difficult=True,perform_FFT=config['data_mode'])


    train_loader, val_loader, test_loader = CreateDataLoaders(dataset,config['dataloader'],config['seed'])

    t_params = sum(p.numel() for p in net.parameters())
    print("Network Parameters: ",t_params)
    print(net)

    # Optimizer
    lr = float(config['optimizer']['lr'])
    step_size = int(config['lr_scheduler']['step_size'])
    gamma = float(config['lr_scheduler']['gamma'])
    num_epochs=int(config['num_epochs'])

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    scheduler_warmup = lr_scheduler.LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=2)
    scheduler_cosine = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs - 2, eta_min=lr * 1e-2)
    scheduler = lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[5])


    print('===========  Optimizer  ==================:')
    print('      LR:', lr)
    print('      step_size:', step_size)
    print('      gamma:', gamma)
    print('      num_epochs:', num_epochs)
    print('')

    # Train
    startEpoch = 0
    global_step = 0
    history = {'train_loss':[],'val_loss':[],'lr':[],'mAP':[],'mAR':[]}
    best_f1 = 0


    if resume:
        print('===========  Resume training  ==================:')
        dict = torch.load(resume)
        net.load_state_dict(dict['net_state_dict'])
        optimizer.load_state_dict(dict['optimizer'])
        scheduler.load_state_dict(dict['scheduler'])
        startEpoch = dict['epoch']+1
        history = dict['history']
        global_step = dict['global_step']

        print('       ... Start at epoch:',startEpoch)


    for epoch in range(startEpoch,num_epochs):

        kbar = pkbar.Kbar(target=len(train_loader), epoch=epoch, num_epochs=num_epochs, width=20, always_stateful=False)

        ###################
        ## Training loop ##
        ###################
        net.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader):

            if config['data_mode'] == 'ADC':
                inputs = data[0].to(device).type(torch.complex64)
            else:
                inputs = data[0].to(device).float()

            label_map = data[1].to(device).float()

            # reset the gradient
            optimizer.zero_grad()

            # forward pass, enable to track our gradient
            with torch.set_grad_enabled(True):
                outputs = net(inputs)


            classif_loss,reg_loss = pixor_loss(outputs['Detection'], label_map,config['losses'])

            classif_loss *= config['losses']['weight'][0]
            reg_loss *= config['losses']['weight'][1]


            loss = classif_loss + reg_loss 

            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.add_scalar('Loss/train_clc', classif_loss.item(), global_step)
            writer.add_scalar('Loss/train_reg', reg_loss.item(), global_step)

            # backprop
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)

            kbar.update(i, values=[("loss", loss.item()), ("class", classif_loss.item()), ("reg", reg_loss.item())])


            global_step += 1


        history['train_loss'].append(running_loss / len(train_loader.dataset))
        scheduler.step()
        history['lr'].append(scheduler.get_last_lr()[0])


        ######################
        ## validation phase ##
        ######################

        eval = run_evaluation(net,val_loader,enc,check_perf=(epoch>=5),
                                detection_loss=pixor_loss,
                                losses_params=config['losses'],config=config, device=device)

        history['val_loss'].append(eval['loss'])
        history['mAP'].append(eval['mAP'])
        history['mAR'].append(eval['mAR'])

        kbar.add(1, values=[("val_loss", eval['loss']),("mAP", eval['mAP']),("mAR", eval['mAR'])])

        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
        writer.add_scalar('Loss/test', eval['loss'], global_step)
        writer.add_scalar('Metrics/mAP', eval['mAP'], global_step)
        writer.add_scalar('Metrics/mAR', eval['mAR'], global_step)

        # Saving all checkpoint as the best checkpoint for multi-task is a balance between both --> up to the user to decide
        name_output_file = config['name']+'_epoch{:02d}_loss_{:.4f}_AP_{:.4f}_AR_{:.4f}.pth'.format(epoch, eval['loss'],eval['mAP'],eval['mAR'])
        filename = os.path.join(output_folder , exp_name , name_output_file)

        #Save best checkpoint based on F1-Score
        if epoch >= 5 and eval['mAP'] + eval['mAR'] > 0 :
            f1_score = 2 * (eval['mAP'] * eval['mAR']) / (eval['mAP'] + eval['mAR'])
            if f1_score > best_f1:
                best_f1 = f1_score
                name_output_file = config['name']+'_AP_{:.4f}_AR_{:.4f}_F1_{:.4f}_best.pth'.format(eval['mAP'], eval['mAR'], f1_score)
                filename = os.path.join(output_folder , exp_name , name_output_file)
                torch.save(checkpoint,filename)

        checkpoint={}
        checkpoint['net_state_dict'] = net.state_dict()
        checkpoint['optimizer'] = optimizer.state_dict()
        checkpoint['scheduler'] = scheduler.state_dict()
        checkpoint['epoch'] = epoch
        checkpoint['history'] = history
        checkpoint['global_step'] = global_step

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

