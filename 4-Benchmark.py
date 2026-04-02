import argparse
import json
import math
import torch
import torch.nn as nn
from dataset.encoder import ra_encoder
from model.MVIT import MViT
from model.RadViT import RadViT
from model.MVIT_ADC import MViT_ADC

import mkl_fft
import numpy as np
import os
import time

class Timer: 
    def __init__(self, name, device):
        self.name = name
        self.device = device
        self.cpu_times = []
        self.gpu_times = []

    def __enter__(self):
        self.cpu_start = time.perf_counter()
        if self.device.type == 'cuda':
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            self.start_event.record()
        return self
    
    def __exit__(self, *args):
        self.cpu_elapsed = (time.perf_counter() - self.cpu_start) * 1000 #ms
        self.cpu_times.append(self.cpu_elapsed)
        if self.device.type == 'cuda':
            self.end_event.record()
            torch.cuda.synchronize()
            self.gpu_elapsed = self.start_event.elapsed_time(self.end_event) #ms
        else:
            self.gpu_elapsed = 0.0
        self.gpu_times.append(self.gpu_elapsed)
        print(f"[{self.name}] CPU: {self.cpu_elapsed:.2f}ms | GPU: {self.gpu_elapsed:.2f}ms")


# Find time T1 Chirp-radar-out

# Radar parameters 
numSamplePerChirp = 512
numChirps = 256

# Build hamming window table to reduce side lobs
hanningWindowRange = (0.54 - 0.46*np.cos(((2*math.pi*np.arange(numSamplePerChirp ))/(numSamplePerChirp -1))))
hanningWindowDoppler = (0.54 - 0.46*np.cos(((2*math.pi*np.arange(numChirps ))/(numChirps -1))))
range_fft_coef = np.expand_dims(np.repeat(np.expand_dims(hanningWindowRange,1), repeats=numChirps, axis=1),2)
doppler_fft_coef = np.expand_dims(np.repeat(np.expand_dims(hanningWindowDoppler, 1).transpose(), repeats=numSamplePerChirp, axis=0),2)


# Step 1: Load ADC signal 
def main(config, root_dir, sample_id, fft=False, fft_shift=False, checkpoint=None):
    timings = {}
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with Timer("1. Load ADC", device) as t:
        adc_signal = os.path.join(root_dir,'ADC_Data',"adc_{:06d}.npy".format(sample_id))
        complex_adc = np.load(adc_signal,allow_pickle=True)
    timings['load_adc'] = t.cpu_elapsed

    parameters = config['model']['vit']

    enc = ra_encoder(geometry = config['dataset']['geometry'],
                        statistics = config['dataset']['statistics'],
                        regression_layer = 2)

    # Step 2: Preprocess the data (FFT, normalization)
    with Timer("2. FFT", device) as t:
        if fft: 
            with Timer("2.1. DC Removal", device) as t1:
                complex_adc = complex_adc - np.mean(complex_adc, axis=(0,1)) # DC removal
            timings['dc_removal'] = t1.cpu_elapsed

            with Timer("2.2. Range FFT", device) as t2:
                range_fft = mkl_fft.fft(np.multiply(complex_adc,range_fft_coef),numSamplePerChirp,axis=0) # Windowing and FFT along range dimension
            timings['range_fft'] = t2.cpu_elapsed

            with Timer("2.3. Doppler FFT", device) as t3:
                input = mkl_fft.fft(np.multiply(range_fft,doppler_fft_coef),numChirps,axis=1) # Windowing and FFT along doppler dimension
            timings['doppler_fft'] = t3.cpu_elapsed


            if fft_shift:
                # Shift doppler zero freqency bin to center of spectrum
                with Timer("2.4. FFT Shift + Split im/re", device) as t4:
                    radar_FFT = np.fft.fftshift(input,axes=1).astype(np.complex64)
                    radar_FFT = np.concatenate([radar_FFT.real,radar_FFT.imag],axis=2)
                timings['fft_shift'] = t4.cpu_elapsed
            else: 
                with Timer("2.4. Split im/re", device) as t4:
                    radar_FFT = np.concatenate([input.real,input.imag],axis=2)
                timings['split_im_re'] = t4.cpu_elapsed

            with Timer("2.5. Normalization", device) as t5:
                if(config['statistics'] is not None): # Normalize the data with the precomputed mean and std values
                        for i in range(len(config['statistics']['input_mean'])):
                            radar_FFT[...,i] -= config['statistics']['input_mean'][i]
                            radar_FFT[...,i] /= config['statistics']['input_std'][i]
            timings['normalization'] = t5.cpu_elapsed
            net = nn.DataParallel(MViT(parameters['D'], parameters['p'], parameters['H'], parameters['W'], parameters['neuron'], parameters['mha'], parameters['layer'], parameters['dropout'], parameters['n_encoders']), device_ids=[0,1,2,3]) 
        else: 
            net = nn.DataParallel(MViT_ADC(parameters['D'], parameters['p'], parameters['H'], parameters['W'], parameters['neuron'], parameters['mha'], parameters['layer'], parameters['dropout'], parameters['n_encoders']), device_ids=[0,1,2,3])
    
    net.to(device)
    dict = torch.load(checkpoint, weights_only=False)
    net.load_state_dict(dict['net_state_dict'])

    # Step 3: Run the model
    with Timer('3. CPU→GPU transfer', device) as t:
        radar_tensor = torch.tensor(radar_FFT).to(device).float()
    timings['transfer'] = t.cpu_elapsed

    print("Warming up GPU...")
    dummy = torch.zeros_like(radar_tensor)
    for _ in range(3):
        with torch.set_grad_enabled(False):
            _ = net(dummy)
    torch.cuda.synchronize()
    print("Warmup done.\n")

    with Timer('4. Network inference', device) as t:
        with torch.set_grad_enabled(False):
            outputs = net(radar_tensor)
    timings['inference'] = t.gpu_elapsed 

    # Step 4: Postprocess the output (decode)


    print("\n=== Résumé des temps ===")
    total = sum(timings.values())
    for name, ms in timings.items():
        print(f"  {name:<30} {ms:>8.2f} ms  ({100*ms/total:.1f}%)")
    print(f"  {'TOTAL':<30} {total:>8.2f} ms")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FFTRadNet Benchmarking')
    parser.add_argument('--root_dir', type=str, default='/path/to/dataset', help='Root directory of the dataset')
    parser.add_argument('--sample_id', type=int, default=0, help='Sample ID to benchmark')
    parser.add_argument('--fft', action='store_true', help='Flag to indicate if FFT should be applied')
    parser.add_argument('--fft_shift', action='store_true', help='Flag to indicate if FFT shift should be applied')
    parser.add_argument('-c', '--config', default='config.json',type=str, help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--checkpoint', default=None, type=str, help='Path to the .pth model checkpoint to resume training')
    args = parser.parse_args()
    config = json.load(open(args.config))
    main(config, args.root_dir, args.sample_id, args.fft, args.fft_shift, args.checkpoint)
