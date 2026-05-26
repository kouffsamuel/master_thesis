import os 
import numpy as np
import cupy as cp
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