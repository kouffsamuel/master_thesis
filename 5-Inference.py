import argparse
import json
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np
import torch
import os
from PIL import Image

from dataset.encoder import ra_encoder
from kmd2_processing.processing import compute_ra, get_complex_content
from utils.metrics import GetDetMetrics
from model.RadViT import RadViT
from utils.metrics import RA_to_cartesian_box, process_predictions_FFT



# ==========================================
# RADAR PARAMETERS
# ==========================================
c = 3e8
fc = 24.125e9
lam = c / fc
BW = 554e6
N = 256
clk = 38461538
delay = 2214#2214

delta_v = (lam * clk * 3.6) / (2 * N * (12 * (N + 4) + delay))
Vmax = delta_v * (N // 2 - 1)

range_bins = np.arange(N) * (c / (2 * BW))
velocity_bins = np.arange(N) * delta_v - Vmax

range_res_out = 0.27498872 * 4
angle_res_out = 0.1 * 4

range_axis = np.arange(128) * range_res_out
angle_axis = (np.arange(224) - 112) * angle_res_out 

def find_closest_index(times_array, target_time):
    return np.argmin(np.abs(times_array - target_time))

def load_files(folder, ext):
    files = sorted(folder.glob(f"*{ext}"))
    times = np.array([float(f.stem[3:]) for f in files])
    return files, times

def load_files_cam(folder, ext):
    files = sorted(folder.glob(f"*{ext}"))
    times = np.array([float(f.stem) for f in files])
    return files, times

fft_files, fft_times = load_files(Path("/Benson_DATA3/Public/MUSE/data_route_2_camionette/RD_shift_with_bg_substraction"), ".npy")
cam_files, cam_times = load_files_cam(Path("/Benson_DATA3/Public/MUSE/data_route_2_camionette/jpeg"), ".jpeg")
output_video = f"/Benson_DATA3/Public/MUSE/data_route_2_camionette/video_inference.mp4"


class RealTimeViewer:
    def __init__(self):
        self.paused = False
        self.fig, (self.ax_rd, self.ax_ra, self.ax_cam) = plt.subplots(1, 3, figsize=(12, 6))

        self.im_rd = None
        self.im_ra = None
        self.im_cam = None
        self.cbar = None
        self.title_rd = self.ax_rd.set_title("")

        # FIX AXES
        self.ax_rd.set_xlim(velocity_bins[0], velocity_bins[-1])
        self.ax_rd.set_ylim(range_bins[0], range_bins[-1])
        self.ax_rd.set_xlabel("Velocity (km/h)")
        self.ax_rd.set_ylabel("Range (m)")

        self.ax_ra.set_xlim(angle_axis[0], angle_axis[-1])
        self.ax_ra.set_ylim(range_axis[0], range_axis[-1])
        self.ax_ra.set_xlabel("Angle (°)")
        self.ax_ra.set_ylabel("Range (m)")

        self.ax_cam.axis("off")

        # KEY PRESS
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        plt.ion()
        plt.show()

    def on_key(self, event):
        if event.key == 'p':
            self.paused = not self.paused
            print("Paused" if self.paused else "Resuming")
        elif event.key == 'q':
            print("Quitting...")
            plt.close('all')
            os._exit(0)

    def update(self, i, enc, net, device):
        fft_file = fft_files[i]
        t = fft_times[i]

        radar_FFT = np.load(fft_file, allow_pickle=True)

        for j in range(len(config['dataset']['statistics']['input_mean'])):
            radar_FFT[...,j] -= config['dataset']['statistics']['input_mean'][j]
            radar_FFT[...,j] /= config['dataset']['statistics']['input_std'][j]
        
        rd_complex = radar_FFT[..., :3] + 1j * radar_FFT[..., 3:]
        rd_map = np.mean(np.abs(rd_complex), axis=2)
        rd_map_db = 20 * np.log10(rd_map + 1e-6)

        if self.im_rd is None:
            self.im_rd = self.ax_rd.imshow(
                rd_map_db.T,
                extent=[velocity_bins[0], velocity_bins[-1], range_bins[0], range_bins[-1]],
                origin='lower',
                cmap='gray_r',
                vmin=0,
                vmax=30,
                aspect='auto'
            )
            self.cbar = self.fig.colorbar(self.im_rd, ax=self.ax_rd)
            self.cbar.set_label("dB")
        else:
            self.im_rd.set_data(rd_map_db.T)

        self.title_rd.set_text(f"Radar t = {t:.6f}")

        # ================= CAMERA =================
        idx = find_closest_index(cam_times, t)
        img = np.array(Image.open(cam_files[idx]))

        if self.im_cam is None:
            self.im_cam = self.ax_cam.imshow(img)
        else:
            self.im_cam.set_data(img)

        self.ax_cam.set_title(f"Camera t = {cam_times[idx]:.6f}")

        # =========== PREDICTION ==============
        radar_FFT = torch.from_numpy(radar_FFT.copy()).float()          # (256, 256, C)
        radar_FFT = radar_FFT.permute(2, 1, 0).unsqueeze(0)      # (1, C, 256, 256)
        radar_FFT = radar_FFT.to(device)

        with torch.set_grad_enabled(False):
            outputs = net(radar_FFT)
        
        out_obj = outputs['Detection'].detach().cpu().numpy().copy()
        conf_map = out_obj[0][0]  # (128, 224)
        if self.im_ra is None:
            self.im_ra = self.ax_ra.imshow(conf_map, aspect='auto', origin='lower', cmap='viridis', extent=[angle_axis[0], angle_axis[-1], range_axis[0], range_axis[-1]] )
            self.ax_ra.set_title("Output Range-Angle")
            self.fig.colorbar(self.im_ra, ax=self.ax_ra, label='Confiance')
        else:
           self.im_ra.set_data(conf_map)
        self.ax_ra.set_title(f"Output Range-Angle t = {t:.6f}")

         # refresh rapide
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        # pause si demandé
        while self.paused:
            plt.pause(0.05)


def main(config):
    """
    Main function to run inference on K-MD2 data and visualize it in real-time.
    Args: 
        config: Configuration dictionary loaded from a JSON file.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    parameters = config['model']['vit']

    net = RadViT(parameters['D'], parameters['p'], parameters['H'], parameters['W'], 
                 parameters['neuron'], parameters['mha'], parameters['layer'], 
                 parameters['dropout'], parameters['n_encoders'], 
                 kmd2=True, 
                 data_mode=config['data_mode'])
    net.to(device)
    
    checkpoint = torch.load("/home/skouff/master_thesis/experiments/RadViT_Class/RD_MVIT_AP_0.8352_AR_0.8299_F1_0.8325_best.pth", weights_only=False, map_location='cpu')
    model_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['net_state_dict'].items()}
    
    net.load_state_dict(model_state_dict)
    net.eval()

    enc = ra_encoder(geometry = config['dataset']['geometry'], 
                    statistics = config['dataset']['statistics'],
                    regression_layer = 2)
    
    viewer = RealTimeViewer()
    writer = FFMpegWriter(fps=15)

    with writer.saving(viewer.fig, output_video, dpi=200):
        for i in range(len(fft_files)):
            viewer.update(i, enc, net, device)
            writer.grab_frame()

            plt.pause(0.001)

    plt.ioff()
    plt.show()
    print("Video saved:", output_video)


    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FFTRadNet Training')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    args = parser.parse_args()

    config = json.load(open(args.config))
    main(config)