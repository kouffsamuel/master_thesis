import os
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

from cfar import CA_CFAR

# ==========================================
# PATHS
# ==========================================
# data_path = "/Benson_DATA3/Public/MUSE/"
data_path = "/DATA_MUSE/"
folder_path = f"{data_path}/2026_05_19_16_35_13/"
save_video = False

# output_video = f"{folder_path}/video_70_80.mp4"
dir_raw = Path(f"{folder_path}/raw")
dir_camera = Path(f"{folder_path}/jpeg")
# background = np.load(f"{data_path}/background_70_80.npy")
# background = np.load(f"{data_path}/background_70_80_raw.npy")
# bg_power = np.load(f"{data_path}/background_puissance.npy")
# save_path = f"{folder_path}/RD_shift_hamming"

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
Vmax = delta_v * (N // 2)

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
        print(f"RX {i} — mean: {sub.mean():.1f}, std: {sub.std():.1f}")
        content[i] = (sub[0::2] + 1j * sub[1::2]).reshape((256, 256))
    return content


def FFT_RD(RX):
    # window_doppler = np.hamming(RX.shape[0])
    # window_range = np.hamming(RX.shape[1])

    # rx = RX * window_doppler[:, None] * window_range[None, :]
    window = np.hamming(256)
    rx = RX * window[:, None]
    fft = np.fft.fft2(RX, axes=(0, 1))
    rd = np.fft.fftshift(fft, axes=0)  # center only Doppler
    return np.abs(rd) #np.abs(rd), garde que reel et imaginaire = 0

def FFT_RA(RX):
    window_range = np.hamming(RX.shape[0])
    window_angle = np.hamming(RX.shape[1])
    rx = RX * window_range[:, None] * window_angle[None, :]
    fft = np.fft.fft2(rx)
    ra = np.fft.fftshift(fft, axes=1)
    return np.abs(ra)

def compute_rd(file):
    content = get_complex_content(file)
    radar_FFT = np.stack([FFT_RD(content[i]) for i in range(3)], axis=2)
    RD_avg = np.mean(radar_FFT, axis=2)
    magn = 20 * np.log10(RD_avg + 1e-6)
    rd_db = np.clip(magn, 0, None)
    return rd_db
    # radar_FFT = np.stack([FFT_RD(content[i]) for i in range(3)], axis=2)

    # rd_power = np.abs(radar_FFT) ** 2                             # puissance du signal
    # rd_power_sub = rd_power / bg_power    # soustraction, pas de négatif
    # rd_amp_sub = np.sqrt(rd_power_sub)                     # nouvelle amplitude

    # # Reconstruction complexe : amplitude corrigée + phase originale conservée
    # rd_sub = rd_amp_sub * np.exp(1j * np.angle(radar_FFT))        # (256, 256, 3) complex

    # radar_FFT = np.concatenate([radar_FFT.real, radar_FFT.imag], axis=2)
    return radar_FFT

def compute_ra(file):
    content = get_complex_content(file)

    window_range = np.hamming(256)
    range_fft = np.fft.fft(content * window_range[None, None, :], axis=2)
    range_fft = range_fft[:, :, :128]  # keep only positive frequencies

    range_profile = np.mean(range_fft, axis=1)

    angle_fft = np.fft.fft(range_profile, n=256, axis=0)  # (n_angle_bins, 128)
    angle_fft = np.fft.fftshift(angle_fft, axes=0)

    ra_map = 20 * np.log10(np.abs(angle_fft) + 1e-6)

    k = np.arange(256) - 128
    sin_theta = np.clip(2 * k / 256, -1, 1)
    angles_deg = np.degrees(np.arcsin(sin_theta))

    return ra_map, angles_deg, range_bins[:128]

# ==========================================
# REAL-TIME VIEWER
# ==========================================
class RealTimeViewer:
    def __init__(self):
        self.paused = False

        self.fig = plt.figure(figsize=(16, 8))
        gs = self.fig.add_gridspec(3, 2, width_ratios=[2, 1], hspace=0.4, wspace=0.3)

        # Radar RD — colonne gauche entière
        self.ax_rd = self.fig.add_subplot(gs[:, 0])

        #Caméra — haut droite
        self.ax_cam = self.fig.add_subplot(gs[:, 1])

        # self.ax_pc = self.fig.add_subplot(gs[:, 0])

        # ADC — 3 lignes droite
        # self.ax_adc = [self.fig.add_subplot(gs[rx, 1]) for rx in range(3)]

        # Init flags
        self.im_rd  = None
        self.im_adc = None
        self.im_cam = None
        self.im_pc  = None
        self.cbar   = None
        self.adc_lines = []

        # ── Radar RD axes ──────────────────────────────────────────────
        self.title_rd = self.ax_rd.set_title("")
        self.ax_rd.set_xlim(velocity_bins[0], velocity_bins[-1])
        self.ax_rd.set_ylim(range_bins[0], range_bins[-1])
        self.ax_rd.set_xlabel("Velocity (km/h)")
        self.ax_rd.set_ylabel("Range (m)")
        
        # self.title_pc = self.ax_pc.set_title("")
        # self.ax_pc.set_xlim(velocity_bins[0], velocity_bins[-1])
        # self.ax_pc.set_ylim(range_bins[0], range_bins[-1])
        # self.ax_pc.set_xlabel("Velocity (km/h)")
        # self.ax_pc.set_ylabel("Range (m)")
        # self.cfar_fonction = CA_CFAR(win_param=(30,14,10,5), threshold=12, rd_size=(256, 256))


        # ── Caméra ─────────────────────────────────────────────────────
        self.ax_cam.axis("off")

        # ── ADC axes ───────────────────────────────────────────────────
        # colors = ['blue', 'red', 'green']
        # for rx in range(3):
        #     self.ax_adc[rx].set_ylabel(f"RX {rx}")
        #     self.ax_adc[rx].grid(True)
        #     self.ax_adc[rx].set_ylim(32000, 33500)
        #     if rx == 0:
        #         self.ax_adc[rx].set_title("ADC signal (I / Q)")
        #     if rx == 2:
        #         self.ax_adc[rx].set_xlabel("Sample index")

        # ── Key press ──────────────────────────────────────────────────
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

        # ── Lecture unique du fichier ───────────────────────────────────
        # adc = get_complex_content(raw_file)
        rd  = compute_rd(raw_file)
        # peaks = self.cfar_fonction(rd).astype(bool)    


        # ================= RADAR RD =================
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

        # if self.im_pc is None:
        #     self.im_pc = self.ax_pc.imshow(
        #         peaks.T,
        #         extent=[velocity_bins[0], velocity_bins[-1], range_bins[0], range_bins[-1]],
        #         origin='lower',
        #         cmap='gray_r',
        #         vmin=0,
        #         vmax=1,
        #         aspect='auto'
        #     )
        #     self.cbar_pc = self.fig.colorbar(self.im_pc, ax=self.ax_pc)
        #     self.cbar_pc.set_label("Hit")
        # else:
        #     self.im_pc.set_data(peaks.T)

        # self.title_pc.set_text(f"Radar t = {t:.6f}")

        # ================= ADC =================
        # colors = ['blue', 'red', 'green']

        # if self.im_adc is None:
        #     for rx in range(3):
        #         chirp0 = adc[rx, 1, :]
        #         li, = self.ax_adc[rx].plot(
        #             np.real(chirp0), color=colors[rx],
        #             linewidth=0.8, label='I'
        #         )
        #         lq, = self.ax_adc[rx].plot(
        #             np.imag(chirp0), color=colors[rx],
        #             linewidth=0.8, linestyle='--', alpha=0.5, label='Q'
        #         )
        #         self.ax_adc[rx].legend(loc='upper right', fontsize=7)
        #         self.adc_lines.append((li, lq))
        #     self.im_adc = True

        # else:
        #     for rx in range(3):
        #         chirp0 = adc[rx, 0, :]
        #         self.adc_lines[rx][0].set_ydata(np.real(chirp0))
        #         self.adc_lines[rx][1].set_ydata(np.imag(chirp0))

        # self.ax_adc[0].set_title(f"ADC t = {t:.6f}")

        # ================= CAMERA =================
        # idx = find_closest_index(cam_times, t)
        # img = np.array(Image.open(cam_files[idx]))

        # if self.im_cam is None:
        #     self.im_cam = self.ax_cam.imshow(img)
        # else:
        #     self.im_cam.set_data(img)

        # self.ax_cam.set_title(f"Camera t = {cam_times[idx]:.6f}")

        # ── Refresh ────────────────────────────────────────────────────
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        while self.paused:
            plt.pause(0.05)

# ==========================================
# MAIN
# ==========================================
def compute_bg_power(raw_files_bg):
    """
    Calcule la puissance moyenne du background sur des frames anéchoïques.
    """
    accum = 0
    for f in raw_files_bg:
        content = get_complex_content(f)
        rd = np.stack([FFT_RD(content[i]) for i in range(3)], axis=2)
        power = np.abs(rd) ** 2
        accum = power if accum is None else accum + power
    return accum / len(raw_files_bg)

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
    # for i in range(len(raw_files)):
    #     print(f"Processing file {i+1}/{len(raw_files)}")
    #     rd = compute_rd(raw_files[i])
    #     np.save(f"{save_path}/rd_{raw_files[i].stem}.npy", rd)
    # contents = []
    # for i in range(len(raw_files)):
    #     print(f"Processing file {i}/{len(raw_files)}")
    #     content = get_complex_content(raw_files[i])
    #     contents.append(content)
    # bg_array = np.mean(contents, axis=0)
    # print(bg_array.dtype)
    # np.save(f"{data_path}/background_70_80_raw.npy", bg_array)
    # raw_files_bg, _ = load_files(Path(f"{data_path}/data_70_80/raw"), ".raw")
    # np.save(f"{data_path}/background_puissance.npy", compute_bg_power(raw_files_bg))


if __name__ == "__main__":
    main()

