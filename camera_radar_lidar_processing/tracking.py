import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
import random

MAX_MISSES = 40
GATE_THRESHOLD = 5

class Tracking:
    """
    A class for tracking objects in the dataset. Based on a Kalman filter for state estimation and the Hungarian algorithm for data association.
    This tracking class is for tracking cluster centroids in the Range-Doppler space to compute SNR and RCS values for the detected objects.
    """
    
    def __init__(self):
        self.tracks = []
        self.track_id = 0
    
    def make_track(self, initial_doppler, initial_range):
        kf = KalmanFilter(dim_x=4, dim_z=2)

        dt = 1

        kf.F = np.array([[1, 0, dt, 0],
                         [0, 1, 0, dt],
                         [0, 0, 1,  0],
                         [0, 0, 0,  1]], dtype=float)

        kf.H = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]], dtype=float)
        
        kf.R = np.diag([0.9, 0.9]) # Confidence in measurements 
        kf.Q = np.diag([0.1, 0.1, 0.5, 0.5]) # Process noise 
        kf.P = np.diag([4.0, 4.0, 5.0, 5.0]) # Initial uncertainty

        kf.x = np.array([[initial_range],
                 [initial_doppler], [0], [0]])

        kf.hits = 1
        kf.misses = 0
        kf.is_confirmed = False
        kf.power_history = []
        kf.centroid_history = []
        kf.track_id = None

        return kf

    
    def extract_clusters(self, detected_bins, labels, rd_matrix, peak_met="mean"):
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

    
    def associate(self, tracks, clusters):
        if len(tracks) == 0:
            return [], [], list(range(len(clusters)))
        
        cost = np.full((len(tracks), len(clusters)),0.0)
        for i, t in enumerate(tracks):
            for j, c in enumerate(clusters):
                cost[i, j] = np.linalg.norm(t.x[:2].flatten() - c["centroid"])
        
        cost[cost > GATE_THRESHOLD] = 1e6
        row_ind, col_ind = linear_sum_assignment(cost)
        matches = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_clusters = list(range(len(clusters)))
        
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < GATE_THRESHOLD:
                matches.append((r, c))
                unmatched_tracks.remove(r)
                unmatched_clusters.remove(c)

        return matches, unmatched_tracks, unmatched_clusters
        
    def step(self, clusters):
        for t in self.tracks:
            t.predict()
        
        matches, unmatched_tracks, unmatched_clusters = self.associate(self.tracks, clusters)

        for t_idx, c_idx in matches:
            track = self.tracks[t_idx]
            c = clusters[c_idx]
            c["track_id"] = track.track_id  

    
            track.update(c["centroid"])
            track.centroid_history.append(c["centroid"].copy())
            track.hits += 1
            track.misses = 0
            if track.hits >= 3:
                track.is_confirmed = True
            track.power_history.append({c["centroid"][0]: c["power"]})
        
        for t_idx in unmatched_tracks:
            self.tracks[t_idx].misses += 1
        
        for c_idx in unmatched_clusters:
            c = clusters[c_idx]
            kf = self.make_track(
                initial_doppler=c["centroid"][1],
                initial_range=c["centroid"][0]
            )
            kf.track_id = self.track_id 
            kf.centroid_history.append(c["centroid"].copy())
            kf.power_history.append({c["centroid"][0]: c["power"]})
            c["track_id"] = kf.track_id
            self.track_id += 1
            self.tracks.append(kf)

        self.tracks = [t for t in self.tracks if t.misses < MAX_MISSES]
    
    def get_confirmed_tracks(self):
        return [t for t in self.tracks if t.is_confirmed]
