import numpy as np
from scipy.ndimage import binary_closing, binary_opening
from processing.radar_parameters import N, velocity_bins, range_bins
from processing.utils import find_closest_index

def get_complex_content(file, N_chirp=256, N_antenna=3):
    data = open(file, "rb").read()
    arr = np.frombuffer(data, dtype=np.uint16)

    content = np.empty((N_antenna, N_chirp, N_chirp), dtype="complex")
    size = 2 * N_chirp * N_chirp
    for i in range(N_antenna):
        sub = arr[i*size:(i+1)*size]
        content[i] = (sub[0::2] + 1j * sub[1::2]).reshape((N_chirp, N_chirp))
    return content

def FFT(RX, N_chirp=256):
    window = np.hamming(N_chirp)
    rx = RX * window[:, None]  # No windowing for now
    fft = np.fft.fft2(RX)
    rd = np.fft.fftshift(fft, axes=0)
    return np.abs(rd) ** 2

def compute_rd(file, background, remove_background=False):
    content = get_complex_content(file)
    RX1 = FFT(content[0])
    RX2 = FFT(content[1])
    RX3 = FFT(content[2])

    RD_avg = (RX1 + RX2 + RX3) / 3
    rd_sub = RD_avg / background if remove_background else RD_avg
    rd = np.clip(rd_sub, 0, None)
    return rd

def extract_clusters(detected_bins, labels, rd_matrix, peak_met="mean"):
    clusters = []

    doppler_bins, range_bins = np.array(detected_bins)


    for k in set(labels):
        if k == -1:
            continue
        mask = (labels == k)
        pts = np.column_stack((doppler_bins[mask], range_bins[mask]))
        r = np.mean(pts[:, 1])
        d = np.mean(pts[:, 0])
        if peak_met == "mean":
            p = np.mean(rd_matrix[pts[:, 0], pts[:, 1]])
        elif peak_met == "max":
            p = np.max(rd_matrix[pts[:, 0], pts[:, 1]])
        elif peak_met == "median":
            p = np.median(rd_matrix[pts[:, 0], pts[:, 1]])

        clusters.append({
            "centroid": np.array([r,d]),
            "power": p,
            "points": pts
        })
    return clusters

def clusterize_radar(rd_power, rd_power_wo, cfar, dbscan, peak_metric="mean"):
    peaks = cfar(rd_power)
    mask = peaks > 0
    # closed = binary_closing(mask, structure=np.ones((5,3))) 
    # opened = binary_opening(closed, structure=np.ones((2,2)))
    detected_bins = np.where(mask > 0)

    # For every peak that is been detected, we need to check if the mirorring peak
    # has a lower amplitude than the main peak. The mirrorring peak is always at 
    # -velocity of current peak and range is at 70m - current range of peak. If it has 
    # lower amplitude we should not include it in the detected_bins.    
    if len(detected_bins[0]) > 0:
        d_idx, r_idx = detected_bins
        n_peaks = len(d_idx)

        bin_to_pos = {
            (int(d_idx[k]), int(r_idx[k])): k for k in range(n_peaks)
        }

        bins_to_remove = set()

        for k in range(n_peaks):
            d_bin = d_idx[k]
            r_bin = r_idx[k]

            mirror_velocity = -velocity_bins[d_bin]
            mirror_range = range_bins[-1] - range_bins[r_bin]

            mirror_d_bin = find_closest_index(velocity_bins, mirror_velocity)
            mirror_r_bin = find_closest_index(range_bins, mirror_range)

            if mirror_d_bin == d_bin and mirror_r_bin == r_bin:
                continue

            mirror_pos = bin_to_pos.get((mirror_d_bin, mirror_r_bin))
            if mirror_pos is None:
                continue

            main_amplitude = rd_power_wo[d_bin, r_bin]
            mirror_amplitude = rd_power_wo[mirror_d_bin, mirror_r_bin]

            if mirror_amplitude < main_amplitude:
                bins_to_remove.add((mirror_d_bin, mirror_r_bin))

        if bins_to_remove:
            detected_bins_filtered = [[], []]
            for d_bin, r_bin in zip(d_idx, r_idx):
                if (int(d_bin), int(r_bin)) in bins_to_remove:
                    continue
                detected_bins_filtered[0].append(d_bin)
                detected_bins_filtered[1].append(r_bin)
        else:
            detected_bins_filtered = [d_idx, r_idx]

        dbscan.fit(np.array(detected_bins_filtered).T)
        labels = dbscan.labels_
        clusters = extract_clusters(
            detected_bins_filtered,
            labels,
            rd_power_wo,
            peak_met=peak_metric,
        )
    else:
        clusters = []

    return clusters, mask
