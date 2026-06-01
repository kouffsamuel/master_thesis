from collections import defaultdict
import os
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from cfar import CA_CFAR
from tracking import Tracking
from sklearn.cluster import DBSCAN, HDBSCAN
from scipy.interpolate import UnivariateSpline


# ==========================================
# PATHS
# ==========================================
# data_path = "/Benson_DATA3/Public/MUSE/"
# folder_path = f"{data_path}/data_route_2_camionette/"
data_path = "/DATA_MUSE/"
folder_path = f"{data_path}/2026_05_20_13_30_38/"
save_video = True

output_video = f"/home/skouff/video.mp4"
dir_raw = Path(f"{folder_path}/radar")
dir_camera = Path(f"{folder_path}/camera")
background = np.load(f"/Benson_DATA3/Public/MUSE/background_db.npy")
background_puissance = np.load(f"/Benson_DATA3/Public/MUSE/background_puissance.npy")


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
        content[i] = (sub[0::2] + 1j * sub[1::2]).reshape((256, 256))
    return content

def FFT(RX):
    window = np.hamming(256)
    rx = RX * window[:, None]
    fft = np.fft.fft2(RX)
    rd = np.fft.fftshift(fft, axes=0)
    return np.abs(rd)


def compute_rd_db(file):
    content = get_complex_content(file)
    RX1 = FFT(content[0])
    RX2 = FFT(content[1])
    RX3 = FFT(content[2])

    RD_avg = (RX1 + RX2 + RX3) / 3
    magn = 20 * np.log10(RD_avg + 1e-6)
    rd_db = np.clip(magn, 0, None)
    return rd_db

def FFT_P(RX):
    window = np.hamming(256)
    rx = RX * window[:, None]
    fft = np.fft.fft2(RX)
    rd = np.fft.fftshift(fft, axes=0)
    return np.abs(rd) ** 2

def compute_rd_power(file, without_background=False):
    content = get_complex_content(file)
    RX1 = FFT_P(content[0])
    RX2 = FFT_P(content[1])
    RX3 = FFT_P(content[2])

    RD_avg = (RX1 + RX2 + RX3) / 3
    rd_sub = RD_avg / background_puissance if without_background else RD_avg
    rd = np.clip(rd_sub, 0, None)
    return rd

def snr_computation():
    snr_per_distances_db = defaultdict(list)
    cfar_fonction = CA_CFAR(win_param=(15,20,9,10), threshold=12, rd_size=(256, 256))
    
    dbscan = DBSCAN(eps=2, min_samples=3)
    # hdbscan = HDBSCAN(
    #     min_cluster_size=3,      
    #     min_samples=2,           
    #     cluster_selection_epsilon=2, 
    #     metric='euclidean'
    # )
    track = Tracking()

    fig, (ax_rd, ax_pc) = plt.subplots(1, 2, figsize=(12, 6))
    plt.ion()
    plt.show()

    p_noise = 10 ** (82.03/10)
    
    for i in range(224,540):
        print(f"Processing frame {i}/{len(raw_files)}")
        rd_power = compute_rd_power(raw_files[i], without_background=True)
        rd_power_wi = compute_rd_power(raw_files[i], without_background=False)


        peaks = cfar_fonction(rd_power)

        detected_bins = np.where(peaks > 0)
        dbscan.fit(np.array(detected_bins).T)
        labels = dbscan.labels_
        clusters = track.extract_clusters(detected_bins, labels, rd_power_wi, peak_met="mean")

        track.step(clusters)
        display_clusters(fig, ax_rd, ax_pc, i, rd_power, peaks, clusters, track)
    
    tracks = track.get_confirmed_tracks()
    best_track = tracks[1]
    for hist in best_track.power_history:
        bin = list(hist.keys())[0]
        bin_corrected = int(np.clip(round(bin), 0, N - 1))
        dist_m = range_bins[bin_corrected]
        p = list(hist.values())[0]
        power_u = p - p_noise
        snr = power_u / p_noise
        snr_db = 10 * np.log10(snr)
        snr_per_distances_db[dist_m].append(snr_db)
    np.save(f"/home/skouff/master_thesis/kmd2_processing/snr_h_13_33_15_new.npy", snr_per_distances_db)

def rcs_computation():
    P_NOISE = 10 ** (82.03/10)
    P_SPHERE_LINEAR = 10 ** (135.1 / 10) - P_NOISE
    SIGMA_SPHERE    = 0.09 * np.pi
    R_SPHERE        = 2.0

    K = SIGMA_SPHERE / (P_SPHERE_LINEAR * (R_SPHERE ** 4))

    rcs_per_distances = defaultdict(list)
    # cfar_fonction = CA_CFAR(win_param=(30,14,20,7), threshold=12, rd_size=(256, 256))
    cfar_fonction = CA_CFAR(win_param=(15,20,9,10), threshold=12, rd_size=(256, 256))
    # cfar_fonction = CA_CFAR(win_param=(14,20,7,10), threshold=12, rd_size=(256, 256))

    
    dbscan = DBSCAN(eps=2, min_samples=3)
    track = Tracking()

    fig, (ax_rd, ax_pc) = plt.subplots(1, 2, figsize=(12, 6))
    plt.ion()
    plt.show()

    for i in range(100,600):
        print(f"Processing frame {i}/{len(raw_files)}")
        rd_power = compute_rd_power(raw_files[i], without_background=True)
        rd_power_with_back = compute_rd_power(raw_files[i], without_background=False)

        peaks = cfar_fonction(rd_power)

        detected_bins = np.where(peaks > 0)
        dbscan.fit(np.array(detected_bins).T)
        labels = dbscan.labels_
        clusters = track.extract_clusters(detected_bins, labels, rd_power_with_back, peak_met="max")

        track.step(clusters)
        display_clusters(fig, ax_rd, ax_pc, i, rd_power, peaks, clusters, track)
    
    tracks = track.get_confirmed_tracks()    
    best_track = tracks[1] 
    for hist in best_track.power_history:
        bin = list(hist.keys())[0]
        bin_corrected = int(np.clip(round(bin), 0, N - 1))
        dist_m = range_bins[bin_corrected]
        p = list(hist.values())[0] 
        power_u = p - P_NOISE
        rcs_per_distances[dist_m].append(power_u)
    np.save(f"/home/skouff/master_thesis/kmd2_processing/rcs_h_13_30_38_n1.npy", rcs_per_distances)

def plot_rcs_snr():
    P_NOISE = 10 ** (82.03/10)
    P_SPHERE_LINEAR = 10 ** (135.1 / 10) - P_NOISE 
    SIGMA_SPHERE    = 0.09 * np.pi
    R_SPHERE        = 2.0

    K = SIGMA_SPHERE / (P_SPHERE_LINEAR * (R_SPHERE ** 4))

    base = f"/home/skouff/master_thesis/kmd2_processing"
    snr_per_distances_db = np.load(f"{base}/snr_h_13_30_38_new_met.npy", allow_pickle=True).item()
    rcs_per_distances    = np.load(f"{base}/rcs_h_13_30_38_n1.npy", allow_pickle=True).item()

    distances  = []
    snr_median = []
    rcs_values = []

    for dist, snr_list in sorted(snr_per_distances_db.items()):
        if len(snr_list) == 0 or dist not in rcs_per_distances:
            continue
        distances.append(dist)
        snr_median.append(np.median(snr_list))
        power_median = np.median(rcs_per_distances[dist])
        rcs_values.append(K * power_median * (dist ** 4))

    distances  = np.array(distances)
    snr_median = np.array(snr_median)
    rcs_values = np.array(rcs_values)

    # Moyenne RCS uniquement sur les zones SNR > 15 dB
    snr_at_distances = np.interp(distances, distances, snr_median)
    mask_reliable    = (snr_at_distances >= 17.0) & (distances <= 50.0)
    rcs_mean         = np.mean(rcs_values[mask_reliable])
    print(f"Mean RCS (SNR ≥ 17 dB) : {rcs_mean:.4f} m²  ({10*np.log10(rcs_mean):.1f} dBsm)")


    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()

    ax1.plot(distances, snr_median, 'r-', lw=2, label='SNR')
    ax2.plot(distances, rcs_values, 'g-', lw=2, label='RCS')

    # # Ligne SNR = 17 dB
    # ax1.axhline(y=17, color='orange', lw=1.5, linestyle='--', label='SNR = 17 dB')

    # # Ligne RCS moyen (zones fiables)
    # ax2.axhline(y=rcs_mean, color='blue', lw=1.5, linestyle='--',
    #             label=f'Mean RCS = {rcs_mean:.3f} m²')

    ax1.set_xlabel('Distance (m)')
    ax1.set_ylabel('SNR (dB)', color='r')
    ax2.set_ylabel('RCS (m²)', color='g')
    ax1.tick_params(axis='y', labelcolor='r')
    ax2.tick_params(axis='y', labelcolor='g')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center')

    ax1.set_title('SNR & RCS vs Distance — Citroën DS4')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.savefig(f"{base}/snr_rcs_cfar_12_12_4_6_10.png")
    plt.show()



def display_clusters(fig, ax_rd, ax_pc, i, rd_power, peaks, clusters, track):
    ax_rd.clear()
    ax_pc.clear()

    ax_rd.imshow(
        10 * np.log10(rd_power).T,
        extent=[velocity_bins[0], velocity_bins[-1], range_bins[0], range_bins[-1]],
        origin="lower",
        cmap="gray_r",
        aspect="auto",
        vmin=0, vmax=30
    )

    ax_pc.imshow(
        peaks.T,
        extent=[velocity_bins[0], velocity_bins[-1], range_bins[0], range_bins[-1]],
        origin="lower",
        cmap="gray_r",
        aspect="auto",
        vmin=0, vmax=1
    )

    cmap = plt.get_cmap("tab20")

    for c in clusters:
        track_id = c.get("track_id", -1)
        color = cmap(track_id % cmap.N) if track_id >= 0 else "red"

        # Afficher tous les bins du cluster
        if "points" in c:
            for (d_bin, r_bin) in c["points"]:
                d_bin = int(np.clip(round(d_bin), 0, N - 1))
                r_bin = int(np.clip(round(r_bin), 0, N - 1))
                ax_pc.scatter(
                    velocity_bins[d_bin], range_bins[r_bin],
                    c=[color], s=5, marker="s", alpha=0.6
                )

        # Afficher le centroïde par dessus avec un X plus grand
        r_bin, d_bin = c["centroid"]
        d_bin = int(np.clip(round(d_bin), 0, N - 1))
        r_bin = int(np.clip(round(r_bin), 0, N - 1))
        ax_pc.scatter(
            velocity_bins[d_bin], range_bins[r_bin],
            c=[color], s=80, marker="x", linewidths=2
        )

    ax_rd.set_xlim(velocity_bins[0], velocity_bins[-1])
    ax_rd.set_ylim(range_bins[0], range_bins[-1])
    ax_rd.set_xlabel("Velocity (km/h)")
    ax_rd.set_ylabel("Range (m)")
    ax_rd.set_title(f"RD frame {i}")

    ax_pc.set_xlim(velocity_bins[0], velocity_bins[-1])
    ax_pc.set_ylim(range_bins[0], range_bins[-1])
    ax_pc.set_xlabel("Velocity (km/h)")
    ax_pc.set_ylabel("Range (m)")
    ax_pc.set_title(f"Clusters frame {i}")

    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.01)



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
        rd_power = compute_rd_power(raw_file, without_background=True)
        rd_db = 10 * np.log10(rd_power)

        if self.im_rd is None:
            self.im_rd = self.ax_rd.imshow(
                rd_db.T,
                extent=[velocity_bins[0], velocity_bins[-1], range_bins[0], range_bins[-1]],
                origin='lower',
                cmap='gray_r',
                vmin=0,
                aspect='auto'
            )
            self.cbar = self.fig.colorbar(self.im_rd, ax=self.ax_rd)
            self.cbar.set_label("dB")
        else:
            self.im_rd.set_data(rd_db.T)

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
    # viewer = RealTimeViewer()
    # writer = FFMpegWriter(fps=15)

    # with writer.saving(viewer.fig, output_video, dpi=200):
    #     for i in range(100, len(raw_files)):
    #         print(f"Processing frame {i}/{len(raw_files)}")
    #         viewer.update(i)
    #         if save_video:
    #             writer.grab_frame()

    #         plt.pause(0.001)

    # plt.ioff()
    # plt.show()
    # print("Video saved:", output_video)
    # snr_computation()
    # rcs_computation()
    plot_rcs_snr()
    
if __name__ == "__main__":
    main()

