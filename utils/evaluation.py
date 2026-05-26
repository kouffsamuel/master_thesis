import torch
import numpy as np
from .metrics import GetFullMetrics, Metrics
import pkbar
import pickle

def run_evaluation(net,loader,encoder,check_perf=False, detection_loss=None,losses_params=None,config=None, device='cuda'):

    metrics = Metrics()
    metrics.reset()
    running_loss = 0.0

    kbar = pkbar.Kbar(target=len(loader), width=20, always_stateful=False)

    for i, data in enumerate(loader):

        # input, out_label,segmap,labels
        if config['data_mode'] == 'ADC':
                inputs = data[0].to(device).type(torch.complex64)
        else:
            inputs = data[0].to(device).float()

        label_map = data[1].to(device).float()

        with torch.set_grad_enabled(False):
            outputs = net(inputs)

        if(detection_loss!=None):
            classif_loss,reg_loss, category_loss = detection_loss(outputs['Detection'], label_map,losses_params)

            classif_loss *= losses_params['weight'][0]
            reg_loss *= losses_params['weight'][1]
            category_loss *= losses_params['weight'][2]

            loss = classif_loss + reg_loss + category_loss

            # statistics
            running_loss += loss.item() * inputs.size(0)

        if(check_perf):
            out_obj = outputs['Detection'].detach().cpu().numpy().copy()
            labels = data[3]

            for pred_obj,true_obj in zip(out_obj,labels):

                metrics.update(ObjectPred=np.asarray(encoder.decode(pred_obj,0.05)),Objectlabels=true_obj,
                            threshold=0.2,range_min=5,range_max=100)

        kbar.update(i)


    mAP,mAR = metrics.GetMetrics()

    return {'loss':running_loss / len(loader.dataset) , 'mAP':mAP, 'mAR':mAR}


def run_FullEvaluation(net,loader,encoder,iou_threshold=0.5,config=None, device='cuda'):

    net.eval()
    results = []
    kbar = pkbar.Kbar(target=len(loader), width=20, always_stateful=False)

    print('Generating Predictions...')
    predictions = {'prediction':{'objects':[],'freespace':[]},'label':{'objects':[],'freespace':[]}}
    for i, data in enumerate(loader):
        
        if config['data_mode'] == 'ADC':
            inputs = data[0].to(device).type(torch.complex64)
        else:
            inputs = data[0].to(device).float()

        with torch.set_grad_enabled(False):
            outputs = net(inputs)

        out_obj = outputs['Detection'].detach().cpu().numpy().copy()

        labels_object = data[3]

        for pred_obj,true_obj in zip(out_obj,labels_object):

            predictions['prediction']['objects'].append( np.asarray(encoder.decode(pred_obj,0.05)))
            predictions['label']['objects'].append(true_obj)

        kbar.update(i)
    # np.save("/home/skouff/master_thesis/predictions_radvit_rd.npy", predictions)
    # predictions = np.load("/home/skouff/master_thesis/predictions_radvit_rd.npy", allow_pickle=True).item()

    iou_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    for iou_ in iou_list:
        results.append(GetFullMetrics(predictions['prediction']['objects'],predictions['label']['objects'],range_min=5,range_max=100,IOU_threshold=iou_))
