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
data_path = "/DATA_MUSE/"
folder_path = f"{data_path}/2026_05_20_13_30_38/"
save_video = True

output_video = f"/home/skouff/video.mp4"
dir_raw = Path(f"{folder_path}/radar")
dir_camera = Path(f"{folder_path}/camera")
background = np.load(f"/Benson_DATA3/Public/MUSE/background_db.npy")
background_puissance = np.load(f"/Benson_DATA3/Public/MUSE/background_puissance.npy")
save_path = Path("/home/skouff/master_thesis/kmd2_processing/")


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


def tracking_and_clustering(save_path, start_frame, end_frame):
    snr_per_distances_db = defaultdict(list)
    rcs_per_distances = defaultdict(list)
    # To be changed according to the object we want to track.
    # Micro-Doppler signatures of pedestrians are typically weaker than those of vehicles.
    cfar_fonction = CA_CFAR(win_param=(15,20,9,10), threshold=12, rd_size=(256, 256))
    
    # eps : The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    # min_samples : The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself.
    dbscan = DBSCAN(eps=2, min_samples=3)
    track_snr = Tracking()
    track_rcs = Tracking()

    fig, (ax_rd, ax_pc_snr, ax_pc_rcs) = plt.subplots(1, 3, figsize=(18, 6))
    plt.ion()
    plt.show()

    p_noise = 10 ** (82.03/10)
    
    for i in range(start_frame, end_frame):
        print(f"Processing frame {i}/{len(raw_files)}")
        rd_power = compute_rd_power(raw_files[i], without_background=True)
        rd_power_wi = compute_rd_power(raw_files[i], without_background=False)

        peaks = cfar_fonction(rd_power)

        detected_bins = np.where(peaks > 0)
        dbscan.fit(np.array(detected_bins).T)
        labels = dbscan.labels_
        # peak_met : metric to compute value of the cluster, can be "mean" or "max"
        clusters = track_snr.extract_clusters(detected_bins, labels, rd_power_wi, peak_met="mean")
        clusters_rcs = track_rcs.extract_clusters(detected_bins, labels, rd_power_wi, peak_met="max")

        track_snr.step(clusters)
        track_rcs.step(clusters_rcs)
        display_clusters(fig, ax_rd, ax_pc_snr, ax_pc_rcs, i, rd_power, peaks, clusters, clusters_rcs)
    
    snr(save_path, snr_per_distances_db, track_snr, p_noise)
    rcs(save_path, rcs_per_distances, track_rcs, p_noise)

def snr(save_path, snr_per_distances_db, track_snr, p_noise):
    tracks_snr = track_snr.get_confirmed_tracks()
    # Can have multiple tracks, we need to identify the one corresponding to our object of interest. 
    # I didn't find a good way to do it, I manually checking whether it was the one with the most hits, but also the one that moved over time.
    best_track = tracks_snr[1]
    for hist in best_track.power_history:
        bin = list(hist.keys())[0]
        bin_corrected = int(np.clip(round(bin), 0, N - 1))
        dist_m = range_bins[bin_corrected]
        p = list(hist.values())[0]
        power_u = p - p_noise
        snr = power_u / p_noise
        snr_db = 10 * np.log10(snr)
        snr_per_distances_db[dist_m].append(snr_db)
    np.save(f"{save_path}/snr.npy", snr_per_distances_db)

def rcs(save_path,rcs_per_distances, track_rcs, p_noise):
    tracks_rcs = track_rcs.get_confirmed_tracks()
    best_track = tracks_rcs[1]
    for hist in best_track.power_history:
        bin = list(hist.keys())[0]
        bin_corrected = int(np.clip(round(bin), 0, N - 1))
        dist_m = range_bins[bin_corrected]
        p = list(hist.values())[0] 
        power_u = p - p_noise
        rcs_per_distances[dist_m].append(power_u)
    np.save(f"{save_path}/rcs.npy", rcs_per_distances)

def plot_rcs_snr(snr_threshold_db=17.0, max_distance=40.0, title="RCS & SNR vs Distance"):
    P_NOISE = 10 ** (82.03/10)
    P_SPHERE_LINEAR = 10 ** (135.1 / 10) - P_NOISE 
    SIGMA_SPHERE    = 0.09 * np.pi
    R_SPHERE        = 2.0

    K = SIGMA_SPHERE / (P_SPHERE_LINEAR * (R_SPHERE ** 4))

    snr_per_distances_db = np.load(f"{save_path}/snr.npy", allow_pickle=True).item()
    rcs_per_distances    = np.load(f"{save_path}/rcs.npy", allow_pickle=True).item()

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

    snr_at_distances = np.interp(distances, distances, snr_median)
    mask_reliable    = (snr_at_distances >= snr_threshold_db) & (distances <= max_distance)
    rcs_mean         = np.mean(rcs_values[mask_reliable])
    print(f"Mean RCS (SNR ≥ {snr_threshold_db} dB) & (Distance ≤ {max_distance} m) : {rcs_mean:.4f} m²  ({10*np.log10(rcs_mean):.1f} dBsm)")


    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()

    ax1.plot(distances, snr_median, 'r-', lw=2, label='SNR')
    ax2.plot(distances, rcs_values, 'g-', lw=2, label='RCS')

    ax1.set_xlabel('Distance (m)')
    ax1.set_ylabel('SNR (dB)', color='r')
    ax2.set_ylabel('RCS (m²)', color='g')
    ax1.tick_params(axis='y', labelcolor='r')
    ax2.tick_params(axis='y', labelcolor='g')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center')

    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}/snr_rcs.png")
    plt.show()



def display_clusters(fig, ax_rd, ax_pc_snr, ax_pc_rcs, i, rd_power, peaks, clusters_snr, clusters_rcs):
    ax_rd.clear()
    ax_pc_snr.clear()
    ax_pc_rcs.clear()

    ax_rd.imshow(
        10 * np.log10(rd_power).T,
        extent=[velocity_bins[0], velocity_bins[-1], range_bins[0], range_bins[-1]],
        origin="lower",
        cmap="gray_r",
        aspect="auto",
        vmin=0, vmax=30
    )

    ax_pc_snr.imshow(
        peaks.T,
        extent=[velocity_bins[0], velocity_bins[-1], range_bins[0], range_bins[-1]],
        origin="lower",
        cmap="gray_r",
        aspect="auto",
        vmin=0, vmax=1
    )

    ax_pc_rcs.imshow(
        peaks.T,
        extent=[velocity_bins[0], velocity_bins[-1], range_bins[0], range_bins[-1]],
        origin="lower",
        cmap="gray_r",
        aspect="auto",
        vmin=0, vmax=1
    )

    cmap = plt.get_cmap("tab20")

    render_clusters(ax_pc_snr, i, clusters_snr, cmap, title="SNR")
    render_clusters(ax_pc_rcs, i, clusters_rcs, cmap, title="RCS")

    ax_rd.set_xlim(velocity_bins[0], velocity_bins[-1])
    ax_rd.set_ylim(range_bins[0], range_bins[-1])
    ax_rd.set_xlabel("Velocity (km/h)")
    ax_rd.set_ylabel("Range (m)")
    ax_rd.set_title(f"RD frame {i}")


    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.01)

def render_clusters(ax_pc, i, clusters, cmap, title="SNR"):
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
    ax_pc.set_xlim(velocity_bins[0], velocity_bins[-1])
    ax_pc.set_ylim(range_bins[0], range_bins[-1])
    ax_pc.set_xlabel("Velocity (km/h)")
    ax_pc.set_ylabel("Range (m)")
    ax_pc.set_title(f"Clusters frame - {title} {i}")



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

    tracking_and_clustering(save_path, start_frame=100, end_frame=600)
    # plot_rcs_snr()
    
if __name__ == "__main__":
    main()

