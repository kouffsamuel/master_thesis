import os 
import numpy as np
import cupy as cp
"""
File for computing the mean and standard deviation of the radar FFT data in the dataset, which will be used for normalization during training.
This Script is specific for K-MD2 data as it not provides regression labels, so it only computes the statistics for the input radar data.
The resulting mean and standard deviation will be printed to the console, and can be used in the data preprocessing steps for training the model.
"""
folder_path_prise_1 = '/Benson_DATA3/Public/MUSE/data_route_2_camionette/RD_shift_hamming'

m = cp.zeros(6)
s = cp.zeros(6)
for file in os.listdir(folder_path_prise_1):
    print(f"Processing file: {file}")
    radar_FFT = np.load(os.path.join(folder_path_prise_1, file), allow_pickle=True)
    
    data = cp.asarray(radar_FFT).reshape(256 * 256, 6)

    m += data.mean(axis=0)
    s += data.std(axis=0)   

print('===  INPUT  ====')
print('mean',m/len(os.listdir(folder_path_prise_1)))
print('std',s/len(os.listdir(folder_path_prise_1)))