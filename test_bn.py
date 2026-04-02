import time

import torch

from dataset.dataset import RADIal
from dataset.dataloader import CreateDataLoaders
from dataset.encoder import ra_encoder
config = {
    "data_mode":"Custom_RD",
    "dataset": {
        "root_dir": "/Benson_DATA3/Public/RADIal/ready_to_use/RADIal/",
        "geometry":{
            "ranges": [512,896,1],
            "resolution": [0.201171875,0.2],
            "size": 3
        },
        "statistics":{
            "input_mean":[-2.62437651e-03, -2.13345470e-01,  1.87890028e-02, -1.44272549e+00,
               -3.76185047e-01,  1.35935890e+00, -2.29865852e-01,  1.22432935e-01,
                1.73593029e+00, -6.53456097e-01,  3.79762385e-01,  5.55212892e+00,
                7.74623837e-01, -1.55886042e+00, -7.24735683e-01,  1.51824262e+00,
               -3.71890112e-01, -8.83323401e-02, -1.61938504e-01,  1.09838836e+00,
                9.99290420e-01, -1.04952120e+00,  1.99716390e+00,  9.28686109e-01,
                1.89906800e+00, -2.37724935e-01,  1.99996778e+00,  7.77374669e-01,
                1.32391419e+00,  1.18174904e+00, -6.96956490e-01,  4.42879647e-01],
            "input_std":[20775.32866954, 23085.38586781, 23017.53041219, 14548.55197203,
                 32133.36398373, 28838.72008902, 27195.74436859, 33103.59882674,
                 32181.40955515, 35022.0350309,  31259.08457192, 36684.42768439,
                 33552.77541718, 25958.69536146, 29532.48663633, 32646.76200793,
                 20728.21451047, 23160.77110769, 23068.97650426, 14915.83084849,
                 32149.49991635, 28958.47224165, 27210.72879388, 33005.5213712,
                 31905.82548318, 35124.76263393, 31258.3364471,  31085.84958294,
                 33628.4113574,  25950.13585943, 29445.21585193, 32885.56853172],
            "reg_mean":[0.4048094369863972,0.3997392847799934],
            "reg_std":[0.6968599580482511,0.6942950877813826]
        }
    },
    "dataloader": {
        "mode":"sequence",
        "split":[0.7,0.15,0.15],
        "train": {
            "batch_size": 4,
            "num_workers": 4
    	},
        "val": {
            "batch_size": 4,
            "num_workers": 4
        },
        "test": {
            "batch_size": 1,
            "num_workers": 1
        }
    },
    "seed":3,
}
enc = ra_encoder(geometry = config['dataset']['geometry'],
                        statistics = config['dataset']['statistics'],
                        regression_layer = 2)

dataset = RADIal(root_dir = config['dataset']['root_dir'],
                        statistics= config['dataset']['statistics'],
                        encoder=enc.encode,
                        difficult=True,perform_FFT=config['data_mode'])


    
train_loader, val_loader, test_loader = CreateDataLoaders(dataset,config['dataloader'],config['seed'])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
for i, data in enumerate(train_loader):
    t0 = time.time()
    # Transfert sur le GPU (adapte selon la structure de ton batch)
    # Exemple si data[0] et data[1] sont des tensors à transférer :
    batch_on_gpu = [d.to(device, non_blocking=True) if torch.is_tensor(d) else d for d in data]
    print(f"Batch {i} loaded+to(GPU) in {time.time() - t0:.3f} sec")
    if i == 20:
        break