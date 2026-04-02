# Charger le checkpoint si nécessaire
import torch

checkpoint = torch.load("/home/skouff/master_thesis/experiments/RD_MVIT_OneEncWithRA/RD_MVIT_AP_0.7573_AR_0.8558_F1_0.8035_best.pth", weights_only=False, map_location='cpu')
history = checkpoint['history']  # contient 'mAP' et 'mAR'

target_map = 0.7572
target_mar = 0.8552

# Trouver le premier époch où la valeur est atteinte ou dépassée
epoch_map = next((i for i, v in enumerate(history['mAP']) if v == target_map), None)
epoch_mar = next((i for i, v in enumerate(history['mAR']) if v == target_mar), None)

print(f"mAP >= {target_map} atteint à l'époch {epoch_map}")
print(f"mAR >= {target_mar} atteint à l'époch {epoch_mar}")