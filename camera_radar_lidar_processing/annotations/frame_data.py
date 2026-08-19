from dataclasses import dataclass, field
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN

from processing.lidar_utils import build_background, clusterize, filter_fov, load_lidar, remove_background
from annotations.lidar_projection import lidar_to_radar_range
from annotations.clusters_matching import  match_radar_clusters_to_lidar
from processing.cfar import CA_CFAR
from processing.radar_processing import compute_rd, clusterize_radar
from processing.radar_parameters import range_bins, N
from processing.utils import find_closest_index
from annotations.master_source import MasterSource

HFOV_CAMERA = 56
VFOV_CAMERA = 33
# HFOV_CAMERA = 64    
# VFOV_CAMERA = 38 

@dataclass
class FrameData:
    t_radar: float
    t_camera: float
    t_lidar: float
    rd_power: np.ndarray
    clusters_radar: list = field(default_factory=list)
    img: np.ndarray = None
    clusters_lidar: list = field(default_factory=list)
    master_detections: list = field(default_factory=list)


class FrameProcessor:
    def __init__(self, master: MasterSource, raw_files, raw_times, cam_files, cam_times, lidar_files, lidar_times, background_radar):
        self.master = master
        self.raw_files = raw_files
        self.raw_times = raw_times

        self.cam_files = cam_files
        self.cam_times = cam_times
        
        self.lidar_files = lidar_files
        self.lidar_times = lidar_times

        self.background_radar = background_radar
     
        self.voxel_occupancy = build_background(lidar_files, n_background=50)
        self.cfar = CA_CFAR(win_param=(12, 12, 4, 6), threshold=10, rd_size=(N, N))
        self.dbscan = DBSCAN(eps=10, min_samples=5)
        self.distance_history = {}

    def process(self, i: int) -> FrameData:
        t_radar = self.raw_times[i]

        rd_power, clusters_radar = self._process_radar(i)
        img, idx_cam = self._load_camera_image(t_radar)
        t_camera = self.cam_times[idx_cam]
        t_lidar, clusters_lidar = self._process_lidar(t_radar)

        context = {"img": img}
        master_detections = self.master.detect(context)

        if clusters_lidar and master_detections:
            clusters_lidar = self.master.match_lidar_clusters(clusters_lidar, master_detections)
            self._update_distance_history(clusters_lidar)

        if clusters_radar and clusters_lidar:
            clusters_radar = match_radar_clusters_to_lidar(
                clusters_lidar, clusters_radar, range_bins, N, max_range_diff_m=2.0
            )

        return FrameData(
            t_radar=t_radar, t_camera=t_camera, t_lidar=t_lidar, rd_power=rd_power,
            clusters_radar=clusters_radar, img=img,
            clusters_lidar=clusters_lidar, master_detections=master_detections,
        )

    def _process_radar(self, i):
        raw_file = self.raw_files[i]
        rd_power = compute_rd(raw_file, background=self.background_radar, remove_background=True)
        rd_power_wo = compute_rd(raw_file, background=self.background_radar, remove_background=False)
        clusters_radar, peaks = clusterize_radar(rd_power, rd_power_wo, self.cfar, self.dbscan)
        return rd_power, clusters_radar

    def _load_camera_image(self, t):
        idx = find_closest_index(self.cam_times, t)
        img = np.array(Image.open(self.cam_files[idx]))
        return img, idx

    def _process_lidar(self, t):
        idx_lidar = find_closest_index(self.lidar_times, t)
        t_lidar = self.lidar_times[idx_lidar]
        pts_raw = load_lidar(self.lidar_files[idx_lidar])
        if pts_raw is None:
            return []
        
        pts = remove_background(pts_raw, self.voxel_occupancy)
        pts = filter_fov(pts, HFOV_CAMERA, VFOV_CAMERA)

        return t_lidar, clusterize(pts)

    def _update_distance_history(self, clusters_lidar):
        for cluster in clusters_lidar:
            if cluster.get("pixel_distance") is None:
                continue
            det_id = cluster["detection_id"]
            self.distance_history.setdefault(det_id, []).append(cluster["pixel_distance"])
            range_meter = lidar_to_radar_range(cluster["center"])
            print(f"ID: {det_id} - distance: {cluster['pixel_distance']}")
            print(f"ID:{det_id} - meter: {range_meter}")