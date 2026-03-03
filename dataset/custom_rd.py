
import math
import math
import os
import mkl_fft
import numpy as np
import pandas as pd

numChirps = 256
numSamplePerChirp = 512
numRxAnt = 16
numTxAnt = 12
hanningWindowRange = (0.54 - 0.46*np.cos(((2*math.pi*np.arange(numSamplePerChirp ))/(numSamplePerChirp -1))))
hanningWindowDoppler = (0.54 - 0.46*np.cos(((2*math.pi*np.arange(numChirps ))/(numChirps -1))))
hanningWindowAzimuth = (0.54 - 0.46*np.cos(((2*math.pi*np.arange(numRxAnt ))/(numRxAnt -1))))
range_fft_coef = np.expand_dims(np.repeat(np.expand_dims(hanningWindowRange,1), repeats=numChirps, axis=1),2)
doppler_fft_coef = np.expand_dims(np.repeat(np.expand_dims(hanningWindowDoppler, 1).transpose(), repeats=numSamplePerChirp, axis=0),2)

root_dir = "/Benson_DATA3/Public/RADIal/ready_to_use/RADIal/"
labels = pd.read_csv(os.path.join(root_dir,'labels.csv')).to_numpy()
unique_ids = np.unique(labels[:,0])
label_dict = {}

for i,ids in enumerate(unique_ids):
    sample_ids = np.where(labels[:,0]==ids)[0]
    label_dict[ids]=sample_ids
sample_keys = list(label_dict.keys())

for index in range(len(sample_keys)):
    sample_id = sample_keys[index]
    radar_name = os.path.join(root_dir,'ADC_Data',"adc_{:06d}.npy".format(sample_id))
    complex_adc = np.load(radar_name,allow_pickle=True)
    complex_adc = complex_adc - np.mean(complex_adc, axis=(0,1))
    range_fft = mkl_fft.fft(np.multiply(complex_adc,range_fft_coef),numSamplePerChirp,axis=0)
    input = mkl_fft.fft(np.multiply(range_fft,doppler_fft_coef),numChirps,axis=1)
    # Shift doppler zero freqency bin to center of spectrum
    radar_FFT = np.fft.fftshift(input,axes=1).astype(np.complex64)
    radar_FFT = np.concatenate([radar_FFT.real,radar_FFT.imag],axis=2)

    out_name = os.path.join(root_dir,'RD_Shift',"fft_{:06d}.npy".format(sample_id))
    np.save(out_name,radar_FFT)
    print("Saved sample {}/{}".format(index+1,len(sample_keys))) 