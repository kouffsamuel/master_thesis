import os
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

# ==========================================
# PATHS
# ==========================================
data_path = "E:"
folder_path = f"{data_path}/rdeth_prise40"
save_video = True

output_video = f"{folder_path}/video.mp4"
dir_raw = Path(f"{folder_path}/raw")
dir_camera = Path(f"{folder_path}/jpeg")
background = np.load(f"{data_path}/background.npy")

# ==========================================
# RADAR PARAMETERS
# ==========================================
c = 3e8
fc = 24.125e9
lam = c / fc
BW = 554e6
N = 256
clk = 38461538
delay = 5415#2214

delta_v = (lam * clk * 3.6) / (2 * N * (12 * (N + 4) + delay))
Vmax = delta_v * (N // 2 - 1)

range_bins = np.arange(N) * (c / (2 * BW))
velocity_bins = np.arange(N) * delta_v - Vmax

# ==========================================
# LOAD FILES + TIMESTAMPS
# ==========================================
def load_files(folder, ext):
    files = sorted(folder.glob(f"*{ext}"))
    times = np.array([float(f.stem) for f in files])
    return files, times

raw_files, raw_times = load_files(dir_raw, ".raw")
cam_files, cam_times = load_files(dir_camera, ".jpeg")

# ==========================================
# UTILS
# ==========================================
def find_closest_index(times_array, target_time):
    return np.argmin(np.abs(times_array - target_time))

# ==========================================
# RADAR PROCESSING
# ==========================================
def get_complex_content(file):
    data = open(file, "rb").read()
    arr = np.frombuffer(data, dtype=np.uint16)

    content = np.empty((3, 256, 256), dtype="complex")
    size = 2 * 256 * 256
    for i in range(3):
        sub = arr[i*size:(i+1)*size]
        content[i] = (sub[0::2] + 1j * sub[1::2]).reshape((256, 256))
    return content

def FFT(RX):
    window = np.hamming(256)
    rx = RX * window[:, None]
    fft = np.fft.fft2(rx)
    rd = np.fft.fftshift(fft, axes=0)
    return np.abs(rd)


def compute_rd(file):
    content = get_complex_content(file)
    RX1 = FFT(content[0])
    RX2 = FFT(content[1])
    RX3 = FFT(content[2])

    RD_avg = (RX1 + RX2 + RX3) / 3
    magn = 20 * np.log10(RD_avg + 1e-6) - background
    rd_db = np.clip(magn, 0, None)
    return rd_db

# ==========================================
# REAL-TIME VIEWER
# ==========================================
class RealTimeViewer:
    def __init__(self):
        self.paused = False
        self.fig, (self.ax_rd, self.ax_cam) = plt.subplots(1, 2, figsize=(12, 6))

        self.im_rd = None
        self.im_cam = None
        self.cbar = None
        self.title_rd = self.ax_rd.set_title("")

        # FIX AXES
        self.ax_rd.set_xlim(velocity_bins[0], velocity_bins[-1])
        self.ax_rd.set_ylim(range_bins[0], range_bins[-1])
        self.ax_rd.set_xlabel("Velocity (km/h)")
        self.ax_rd.set_ylabel("Range (m)")

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

    def update(self, i):
        raw_file = raw_files[i]
        t = raw_times[i]

        # ================= RADAR =================
        rd = compute_rd(raw_file)

        if self.im_rd is None:
            self.im_rd = self.ax_rd.imshow(
                rd.T,
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
            self.im_rd.set_data(rd.T)

        self.title_rd.set_text(f"Radar t = {t:.6f}")

        # ================= CAMERA =================
        idx = find_closest_index(cam_times, t)
        img = np.array(Image.open(cam_files[idx]))

        if self.im_cam is None:
            self.im_cam = self.ax_cam.imshow(img)
        else:
            self.im_cam.set_data(img)

        self.ax_cam.set_title(f"Camera t = {cam_times[idx]:.6f}")

        # refresh rapide
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        # pause si demandé
        while self.paused:
            plt.pause(0.05)

# ==========================================
# MAIN
# ==========================================
def main():
    viewer = RealTimeViewer()
    writer = FFMpegWriter(fps=15)

    with writer.saving(viewer.fig, output_video, dpi=200):
        for i in range(len(raw_files)):
            viewer.update(i)
            if save_video:
                writer.grab_frame()

            plt.pause(0.001)

    plt.ioff()
    plt.show()
    print("Video saved:", output_video)

if __name__ == "__main__":
    main()

