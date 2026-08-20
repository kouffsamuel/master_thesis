import os
from annotations.frame_data import FrameData
from pathlib import Path
import numpy as np
from processing.radar_parameters import velocity_bins, range_bins, N
import matplotlib.pyplot as plt
import open3d as o3d

class ProcessedDataRecorder:
    def __init__(self, output_path, folder_name):
        self.folder_name = folder_name
        self.root_dir = Path(f"{output_path}/{folder_name}")
        for sensor in ["camera", "radar", "lidar"]:
            (self.root_dir / sensor).mkdir(parents=True, exist_ok=True)        

    def update(self, frame:FrameData):
        rd_power = 10 * np.log10(frame.rd_power)
        img = frame.img
        clusters_lidar = frame.clusters_lidar   

        # Camera
        camera_path = os.path.join(self.root_dir, "camera", f"{frame.t_camera}.jpeg")
        plt.imsave(camera_path, img)

        # Radar 
        radar_path = os.path.join(self.root_dir, "radar", f"{frame.t_radar}.raw")
        rd_power.astype(np.float32).tofile(radar_path)

        # Lidar
        lidar_path = os.path.join(self.root_dir, "lidar", f"{frame.t_lidar}.ply")
        point_cloud = o3d.geometry.PointCloud()

        for cluster in clusters_lidar:
            point_cloud += cluster['pcd']

        o3d.io.write_point_cloud(str(lidar_path),point_cloud)



