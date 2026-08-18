import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def load_files(folder, ext):
    files = sorted(folder.glob(f"*{ext}"))
    times = np.array([float(f.stem) for f in files])
    return files, times
    
    
def find_closest_index(times_array, target_time):
    return np.argmin(np.abs(times_array - target_time))
    
    
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

    # Moyenne RCS uniquement sur les zones SNR > 15 dB
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