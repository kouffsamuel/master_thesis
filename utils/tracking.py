import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
import motmetrics as mm
import random

class Tracker:
    """
    Tracker represents a single tracked object using a Kalman filter to estimate its state (position and velocity).
    It maintains the track's ID, the Kalman filter instance, the bounding box corners, the number of hits, and the number of consecutive misses.
    The Tracker can predict the next state, update its state with new detections, and confirm its ID after a certain number of hits.
    """
    _id_counter = 0
    
    def __init__(self, centroid, box_corners):
        self.id = None
        self.kf = self.make_track(centroid)
        self.box_corners = box_corners
        self.hits = 1
        self.no_match = 0
    
    def make_track(self, centroid):
        kf = KalmanFilter(dim_x=4, dim_z=2)

        dt = 0.2 # Time step (5 Hz)

        kf.F = np.array([[1, 0, dt, 0],
                         [0, 1, 0, dt],
                         [0, 0, 1,  0],
                         [0, 0, 0,  1]], dtype=float)

        kf.H = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]], dtype=float)
        
        kf.R = np.diag([8.0, 8.0]) # Confidence in measurements 
        kf.Q = np.diag([0.1, 0.1, 1.0, 1.0]) # Process noise 
        kf.P = np.diag([4.0, 4.0, 5.0, 5.0]) # Initial uncertainty

        kf.x[:2] = centroid.reshape(2, 1)

        return kf
    
    def confirm(self):
        if self.id is None:
            self.id = Tracker._id_counter
            Tracker._id_counter += 1
    
    def predict(self):
        self.kf.predict()
    
    def update(self, centroid, box_corners):
        predicted = self.kf.x[:2].flatten().copy()
        self.kf.update(centroid.reshape(2, 1))
        residual = centroid - predicted
        print(f"ID {self.id} | residual: {residual} m")
        self.box_corners = box_corners
        self.hits += 1
        self.no_match = 0
    
    @property
    def predicted_centroid(self):
        return self.kf.x[:2].flatten()
    
class MultiObjectTracker:
    """
    MultiObjectTracker manages multiple Tracker instances to maintain object identities across frames.
    It uses the Hungarian algorithm for data association based on Mahalanobis distance and prunes tracks 
    that have not been matched for a certain number of frames.
    """
    def __init__(self, max_misses, min_hits, mahal_threshold):
        self.tracks = []
        self.max_misses = max_misses
        self.min_hits = min_hits
        self.mahal_threshold = mahal_threshold

    def _prune(self):
        self.tracks = [t for t in self.tracks if t.no_match <= self.max_misses]
    def _active(self):
        return [
            {
                'id'      : t.id,
                'centroid': t.predicted_centroid,
                'corners' : t.box_corners,
                'hits'    : t.hits,
            }
            for t in self.tracks
            if t.hits >= self.min_hits  and t.id is not None
        ]
    
    @staticmethod
    def _mahalanobis_cost(tracks, det_centroids):
        """
        Compute the Mahalanobis distance cost matrix between predicted track centroids and detected centroids.
        The cost is set to a high value (1e8) for pairs that are physically implausible (beyond max_euclidian distance).
        Written with help of Claude.ai
        Args:
            tracks: List of Tracker objects representing the current tracks with their Kalman filter states.
            det_centroids: Numpy array of shape (M, 2) containing the centroids of the current detections.
        """
        N, M = len(tracks), len(det_centroids)
        cost  = np.full((N, M), 1e8)  # Coût élevé par défaut
        v_max = 148 / 3.6
        max_euclidian = v_max * 0.2 * 1.5 # 20% de la vitesse max en 0.2s, x2 pour être large
        for i, t in enumerate(tracks):
            # Covariance de la prédiction (position uniquement)
            S = t.kf.H @ t.kf.P @ t.kf.H.T + t.kf.R  # (2x2)
            S_inv = np.linalg.inv(S)
            mu = t.predicted_centroid  # (2,)

            for j, det in enumerate(det_centroids):
                diff = det - mu
                if np.linalg.norm(diff) > max_euclidian:
                    continue  # Ignorer si la détection est trop loin (hors portée physique)
                cost[i, j] = diff @ S_inv @ diff  # distance de Mahalanobis²
        return cost  
    
    def update(self, detections):
        for t in self.tracks:
            t.predict()
        
        if len(detections) == 0:
            for t in self.tracks:
                t.no_match += 1
            self._prune()
            return self._active()
        else: 
            detections = detections[:, 1:9]
        
        det_corners = detections.reshape(-1, 4, 2)
        det_centroids = np.array([c.mean(axis=0) for c in det_corners])

        if len(self.tracks) == 0:
            for i in range(len(detections)):
                self.tracks.append(Tracker(det_centroids[i], det_corners[i]))
            return self._active()

        pred_centroids = np.array([t.predicted_centroid for t in self.tracks])
        cost_matrix = self._mahalanobis_cost(self.tracks, det_centroids)

        row_idx, col_idx = linear_sum_assignment(cost_matrix)

        matched_tracks = set()
        matched_dets = set()

        for r, c in zip(row_idx, col_idx):
            if cost_matrix[r, c] <= self.mahal_threshold:
                self.tracks[r].update(det_centroids[c], det_corners[c])
                matched_tracks.add(r)
                matched_dets.add(c)
        
        for r, t in enumerate(self.tracks):
            if r not in matched_tracks:
                t.no_match += 1
            if r in matched_tracks and t.hits >= self.min_hits:
                t.confirm()
        
        for c in range(len(detections)):
            if c not in matched_dets:
                self.tracks.append(Tracker(det_centroids[c], det_corners[c]))
        
        self._prune()
        return self._active()

class MOTEvaluator:
    """
    MOTEvaluator uses the motmetrics library to evaluate tracking performance.
    It maintains a MOTAccumulator to accumulate tracking results over frames and computes metrics at the end.
    """
    def __init__(self):
        self.acc = mm.MOTAccumulator(auto_id=False)
        v_max = 148 / 3.6
        self.max_euclidian = v_max * 0.2 * 1.5
    
    def update(self, gt_frame, active_tracks, frame_idx):
        gt_centroids = gt_frame[['cx', 'cy']].values
        gt_ids = gt_frame['track_id'].values
        
        if len(active_tracks) > 0:
            pred_centroids = np.array([t['centroid'] for t in active_tracks])
            pred_ids = np.array([t['id'] for t in active_tracks])
        else:
            pred_centroids = np.empty((0, 2))
            pred_ids = []
        
        if len(pred_centroids) > 0:
            C = np.linalg.norm(gt_centroids[None, :] - pred_centroids[:, None], axis=-1)
            C[C > self.max_euclidian] = np.nan
        else:
            C = np.full((len(gt_ids), 0), np.nan)
        self.acc.update(gt_ids, pred_ids, C, frameid=frame_idx)
    
    def summary(self, all_accs, SEQUENCES):
        mh = mm.metrics.create()
        summary = mh.compute_many(
            all_accs,
            metrics=mm.metrics.motchallenge_metrics,
            names=SEQUENCES
        )
        print(mm.io.render_summary(summary, formatters=mh.formatters))

        






