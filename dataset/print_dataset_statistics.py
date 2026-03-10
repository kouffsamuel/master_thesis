import sys
sys.path.insert(0, '../')

from dataset.dataset import RADIal
from dataset.encoder import rd_encoder

import numpy as np


geometry = {    "ranges": [512,256],
                "resolution": [0.201171875,0.1],
                "size": 3}

statistics = {  "input_mean":np.zeros(32),
                "input_std":np.ones(32),
                "reg_mean":np.zeros(3),
                "reg_std":np.ones(3)}

# Can also use ra_encoder, just replace rd_encoder with ra_encoder
enc = rd_encoder(geometry = geometry,
                    statistics = statistics,
                    regression_layer = 2)

dataset = RADIal(root_dir = "/Benson_DATA3/Public/RADIal/ready_to_use/RADIal",
                       statistics=None,
                       encoder=enc.encode,
                       difficult=True,perform_FFT='Custom_RD')

reg = []
m=0
s=0
for i in range(len(dataset)):
    print(i,len(dataset))
    radar_FFT, segmap,out_label,box_labels, _= dataset.__getitem__(i)

    data = np.reshape(radar_FFT,(512*256,32))

    m += data.mean(axis=0)
    s += data.std(axis=0)

    idy,idx = np.where(out_label[0]>0)

    reg.append(out_label[1:,idy,idx])

reg = np.concatenate(reg,axis=1)

print('===  INPUT  ====')
print('mean',m/len(dataset))
print('std',s/len(dataset))

print('===  Regression  ====')
print('mean',np.mean(reg,axis=1))
print('std',np.std(reg,axis=1))
