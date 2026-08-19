from annotations.frame_data import FrameData
import numpy as np
from processing.radar_parameters import range_bins, velocity_bins, N
import json

class DetectionRecorder:
    def __init__(self, folder_name: str):
        self.folder_name = folder_name
        self.json_data = {}

    def record_frame(self, frame: FrameData):
        if not (frame.master_detections or frame.clusters_lidar or frame.clusters_radar):
            return  # frame totalement vide, rien à enregistrer

        camera_entries = [
            {
                "id": det["id"],
                "cx": det.get("center", (None, None))[0],
                "cy": det.get("center", (None, None))[1],
                "width": det.get("width"),
                "height": det.get("height"),
                "class": det.get("class")
            }
            for det in frame.master_detections
        ]

        lidar_entries = []
        for cluster in frame.clusters_lidar:
            center = cluster["center"]
            lidar_entries.append({
                "detection_id": cluster.get("detection_id"),
                "x_m": float(center[0]),
                "y_m": float(center[1]),
                "z_m": float(center[2]),
            })

        radar_entries = []
        for cluster in frame.clusters_radar:
            r_bin, d_bin = cluster["centroid"]
            r_bin = int(np.clip(round(r_bin), 0, N - 1))
            d_bin = int(np.clip(round(d_bin), 0, N - 1))
            radar_entries.append({
                "detection_id": cluster.get("detection_id"), 
                "radar_m": float(range_bins[r_bin]),
                "radar_mps": float(velocity_bins[d_bin]) / 3.6,
            })

        self.json_data.setdefault(self.folder_name, {})[str(frame.t_radar)] = {
            "t_radar": frame.t_radar,
            "t_camera": frame.t_camera,
            "t_lidar": frame.t_lidar,
            "camera_detections": camera_entries,
            "lidar_clusters": lidar_entries,
            "radar_clusters": radar_entries,
        }

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.json_data, f, indent=2)
        print("JSON saved:", path)