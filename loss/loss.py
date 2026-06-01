import torch
import torch.nn.functional as F
import torch.nn as nn
"""
loss.py file from FFTRadNet
"""
class FocalLoss(nn.Module):
    """
    Focal loss class. Stabilize training by reducing the weight of easily classified background sample and focussing
    on difficult foreground detections.
    """

    def __init__(self, gamma=0, size_average=False):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, prediction, target):

        # get class probability
        pt = torch.where(target == 1.0, prediction, 1-prediction)

        # compute focal loss
        loss = -1 * (1-pt)**self.gamma * torch.log(pt+1e-6)

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
    
def pixor_loss(batch_predictions, batch_labels,param, device='cuda'):

    #########################
    #  classification loss  #
    #########################
    classification_prediction = batch_predictions[:, 0,:, :].contiguous().flatten()
    classification_label = batch_labels[:, 0,:, :].contiguous().flatten()

    if(param['classification']=='FocalLoss'):
        focal_loss = FocalLoss(gamma=2)
        classification_loss = focal_loss(classification_prediction, classification_label)
    else:
        classification_loss = F.binary_cross_entropy(classification_prediction.double(), classification_label.double(),reduction='sum')

    
    #####################
    #  Regression loss  #
    #####################

    regression_prediction = batch_predictions.permute([0, 2, 3, 1])[:, :, :, :-1]
    regression_prediction = regression_prediction.contiguous().view([regression_prediction.size(0)*
                        regression_prediction.size(1)*regression_prediction.size(2), regression_prediction.size(3)])
    regression_label = batch_labels.permute([0, 2, 3, 1])[:, :, :, :-1]
    regression_label = regression_label.contiguous().view([regression_label.size(0)*regression_label.size(1)*
                                                           regression_label.size(2), regression_label.size(3)])

    positive_mask = torch.nonzero(torch.sum(torch.abs(regression_label), dim=1))
    pos_regression_label = regression_label[positive_mask.squeeze(), :]
    pos_regression_prediction = regression_prediction[positive_mask.squeeze(), :]


    T = batch_labels[:,1:3]
    P = batch_predictions[:,1:3]
    M = batch_labels[:,0].unsqueeze(1)

    if(param['regression']=='SmoothL1Loss'):
        reg_loss_fct = nn.SmoothL1Loss(reduction='sum')
    else:
        reg_loss_fct = nn.L1Loss(reduction='sum')
    
    regression_loss = reg_loss_fct(P*M,T)
    NbPts = M.sum()
    if(NbPts>0):
        regression_loss/=NbPts

    ########################
    #  Vehicle class loss  #
    ########################
    cat_logits = batch_predictions[:, 3:, :, :]
    cat_labels = batch_labels[:, 3, :, :]

    pos_mask = (M.squeeze(1) > 0) & (cat_labels >= 0)
    
    if pos_mask.any():
        logits = cat_logits.permute(0, 2, 3, 1)[pos_mask]
        targets = cat_labels[pos_mask].long()
        class_weights = torch.tensor([1.0, 4.87, 0.0, 0.0, 0.0], device=device)
        category_loss = F.cross_entropy(logits, targets, weight=class_weights, reduction='sum') / pos_mask.sum()
    else:
        category_loss = torch.tensor(0.0, device=batch_predictions.device)

    return classification_loss,regression_loss,category_loss


def detection_loss(output, box_labels, image_size):
    predictions = output["Detection"]
    W,H = image_size

    device = predictions.device
    BATCH_SIZE = predictions.shape[0]
    total_class_loss = 0
    total_reg_loss = 0

    for i in range(BATCH_SIZE):
        pred = predictions[i]
        pred_boxes = pred[:, 1:]
        pred_conf = pred[:, 0]

        raw = box_labels[i].to(device)

        if len(raw) == 0:
            total_class_loss += F.binary_cross_entropy(pred_conf, torch.zeros_like(pred_conf), reduction='sum')
            continue

        if raw.dim() == 1:
            raw = raw.unsqueeze(0)
        
        # Ground truth range / doppler bins (normalised by image size)
        r_bin = raw[:, 1]
        d_bin = raw[:, 2]
        gt_boxes = torch.stack([r_bin / W, d_bin / H], dim=1)

        num_gt = gt_boxes.shape[0]

        # Pairwise L1 distance between all predictions and GT boxes
        with torch.no_grad():
            dist = torch.cdist(pred_boxes, gt_boxes, p=1)  # (N_pred, N_gt)

            # Greedy 1-1 matching: at chaque étape, on prend la paire la plus proche
            num_matches = min(dist.shape[0], dist.shape[1])
            matched_pred_idx = []
            matched_gt_idx = []

            dist_work = dist.clone()
            for _ in range(num_matches):
                min_val, flat_idx = torch.min(dist_work.view(-1), dim=0)
                pred_idx = (flat_idx // num_gt).item()
                gt_idx = (flat_idx % num_gt).item()

                matched_pred_idx.append(pred_idx)
                matched_gt_idx.append(gt_idx)

                # Invalider cette ligne et cette colonne pour garder l'assignation 1-1
                dist_work[pred_idx, :] = float('inf')
                dist_work[:, gt_idx] = float('inf')

            matched_pred_idx = torch.tensor(matched_pred_idx, device=device, dtype=torch.long)
            matched_gt_idx = torch.tensor(matched_gt_idx, device=device, dtype=torch.long)

        # Regression loss sur les paires appariées
        total_reg_loss += F.l1_loss(pred_boxes[matched_pred_idx], gt_boxes[matched_gt_idx], reduction='sum')

        # Classification: 1 pour les prédictions appariées, 0 pour le reste
        gt_conf = torch.zeros_like(pred_conf)
        gt_conf[matched_pred_idx] = 1.0

        total_class_loss += F.binary_cross_entropy(pred_conf, gt_conf, reduction='sum')
    
    return total_class_loss/BATCH_SIZE, total_reg_loss/BATCH_SIZE






    