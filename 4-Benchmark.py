import argparse
import json
import math
import torch
import torch.nn as nn
from dataset.encoder import ra_encoder
from model.RadViT import RadViT
from thop import profile 
from thop import clever_format
from model.fourier_net import FFT_Net


import mkl_fft
import numpy as np
import os
import time

class Timer: 
    """
    Context manager to measure CPU and GPU time for a block of code. 
    Written with help from Claude.ai.
    """
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


# Find time T1 Chirp-radar-out # 24ms
WARMUP = False

# Radar parameters 
numSamplePerChirp = 512
numChirps = 256

# Build hamming window table to reduce side lobs
hanningWindowRange = (0.54 - 0.46*np.cos(((2*math.pi*np.arange(numSamplePerChirp ))/(numSamplePerChirp -1))))
hanningWindowDoppler = (0.54 - 0.46*np.cos(((2*math.pi*np.arange(numChirps ))/(numChirps -1))))
range_fft_coef = np.expand_dims(np.repeat(np.expand_dims(hanningWindowRange,1), repeats=numChirps, axis=1),2)
doppler_fft_coef = np.expand_dims(np.repeat(np.expand_dims(hanningWindowDoppler, 1).transpose(), repeats=numSamplePerChirp, axis=0),2)


# Step 1: Load ADC signal 
def main(config, sample_id, device='cuda'):
    """
    Main function to benchmark the inference time of the model on a single sample.
    Written with help from Claude.ai.
    Args:
        config: Configuration dictionary loaded from a JSON file.
        sample_id: ID of the sample to benchmark.
        device: Device to run the benchmark on.
    """
    timings = {}
    with Timer("1. Load ADC", device) as t:
        adc_signal = os.path.join(config['dataset']['root_dir'],'ADC_Data',"adc_{:06d}.npy".format(sample_id))
        complex_adc = np.load(adc_signal,allow_pickle=True)
    timings['load_adc'] = t.cpu_elapsed


    # Step 2: Preprocess the data (FFT, normalization)
    if config['data_mode'] != 'ADC':
        with Timer("2. FFT", device) as t:
            if config['data_mode'] == "Custom_RD" or config['data_mode'] == "RD": 
                with Timer("2.1. DC Removal", device) as t1:
                    complex_adc = complex_adc - np.mean(complex_adc, axis=(0,1)) # DC removal
                timings['dc_removal'] = t1.cpu_elapsed

                with Timer("2.2. Range FFT", device) as t2:
                    range_fft = mkl_fft.fft(np.multiply(complex_adc,range_fft_coef),numSamplePerChirp,axis=0) # Windowing and FFT along range dimension
                timings['range_fft'] = t2.cpu_elapsed

                with Timer("2.3. Doppler FFT", device) as t3:
                    input = mkl_fft.fft(np.multiply(range_fft,doppler_fft_coef),numChirps,axis=1) # Windowing and FFT along doppler dimension
                timings['doppler_fft'] = t3.cpu_elapsed


                if config['data_mode'] == "Custom_RD":
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
                    if(config["dataset"]['statistics'] is not None): # Normalize the data with the precomputed mean and std values
                            mean = np.array(config['dataset']['statistics']['input_mean'])  # (16,)
                            std = np.array(config['dataset']['statistics']['input_std'])    # (16,)
                            radar_FFT = (radar_FFT - mean) / std  # broadcasting automatique
                timings['normalization'] = t5.cpu_elapsed

    # Step 3: Run the model
    with Timer('3. CPU→GPU transfer', device) as t:
        if config['data_mode'] == 'ADC':
            radar_tensor = torch.tensor(complex_adc).permute(2, 0, 1).unsqueeze(0).to(device).type(torch.complex64)
        else:
            radar_tensor = torch.tensor(radar_FFT).permute(2, 0, 1).unsqueeze(0).to(device).float()
    timings['transfer'] = t.cpu_elapsed

    with Timer('4. Network inference', device) as t:
        with torch.set_grad_enabled(False):
            outputs = net(radar_tensor)
    timings['inference'] = t.gpu_elapsed 

    # Step 4: Postprocess the output (decode)
    out_obj = outputs['Detection'].detach().cpu().numpy().copy()
    encoder = ra_encoder(geometry = config['dataset']['geometry'], statistics = config['dataset']['statistics'], regression_layer = 2)
    decoded_objects = []
    for pred_obj in out_obj:
        with Timer('5. Decode output', device) as t:
            decoded_object = encoder.decode(pred_obj, threshold=0.05)
            decoded_objects.append(decoded_object)
        timings['decode'] = t.cpu_elapsed


    print("\n=== Résumé des temps ===")
    total = sum(timings.values())
    for name, ms in timings.items():
        print(f"  {name:<30} {ms:>8.2f} ms  ({100*ms/total:.1f}%)")
    print(f"  {'TOTAL':<30} {total:>8.2f} ms")
    return timings

def fft_ops_counter(module, input):
    """
    Custom FLOPs counter for the FFT_Net module.
    Claude.ai generated
    Args:
        module: The module for which to count FLOPs (should be an instance of FFT_Net).
        input: The input tensor(s) to the module.

    """
    N = input[0].shape[-1]
    num_ffts = input[0].numel() // N
    module.total_ops += torch.DoubleTensor([5 * N * math.log2(N) * num_ffts])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FFTRadNet Benchmarking')
    parser.add_argument('-c', '--config', default='config.json',type=str, help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--checkpoint', default=None, type=str, help='Path to the .pth model checkpoint to resume training')
    args = parser.parse_args()
    
    config = json.load(open(args.config))
    
    timings_list = []
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    parameters = config['model']['vit']
    net = nn.DataParallel(RadViT(parameters['D'], parameters['p'], parameters['H'], 
                                 parameters['W'], parameters['neuron'], parameters['mha'], 
                                 parameters['layer'], parameters['dropout'], 
                                 parameters['n_encoders'], data_mode=config['data_mode']), device_ids=[0,1,2])
    
    dict = torch.load(args.checkpoint, map_location=device, weights_only=False)
    net.load_state_dict(dict['net_state_dict'])
    net.to(device)

    print("Warming up GPU...")
    if config['data_mode'] == 'ADC':
        dummy = torch.zeros((1, 16, numSamplePerChirp, numChirps)).to(device).type(torch.complex64)
    else:
        dummy = torch.zeros((1, 32, numSamplePerChirp, numChirps)).to(device).float()

    for _ in range(10):
        with torch.set_grad_enabled(False):
            _ = net(dummy)
    torch.cuda.synchronize()

    print("Warmup done.\n")
    print("Computing FLOPs and parameters...")
    macs, params = profile(net.module, inputs=(dummy,), verbose=False, custom_ops={FFT_Net: fft_ops_counter})
    flops = 2 * macs  # FLOPs = 2 * MACs
    flops_str, params_str = clever_format([flops, params], "%.3f")
    print(f"Model FLOPs: {flops_str} | Parameters: {params_str}\n")

    for i, file in enumerate(os.listdir(config['dataset']['root_dir'] + '/ADC_Data')):
        if file.endswith('.npy'):
            sample_id = int(file.split('_')[1].split('.')[0])
            print(f"\n=== Benchmarking sample {sample_id} ===")
            timings = main(config, sample_id, device)
            timings_list.append(timings)
        if i >= 99: # Limit to 100 samples for benchmarking
            break
    
    # Compute average timings
    keys = timings_list[0].keys()
    for key in keys:
        values = [t[key] for t in timings_list if key in t]
        print(f"  {key:<30} {np.mean(values):>8.2f} ms ± {np.std(values):.2f} ms")
    
   
