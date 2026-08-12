import argparse
from PIL import Image

from utils import find_closest_index
from real_time_viewer import RealTimeViewer
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from pathlib import Path
import numpy as np
from radar_processing import compute_rd
from lidar_processing import load_lidar

data_path = "/DATA_MUSE"
folder_path = f"{data_path}/2026_08_07_10_05_00/"

# folder_path = f"{data_path}/2026_07_23_17_22_04/" # Avec plexi
# folder_path = f"{data_path}/2026_07_23_17_28_05/" # Sans plexi

save_video = True

output_video = f"/home/skouff/video_test.mp4"
# dir_raw = Path(f"{folder_path}/raw")
# dir_camera = Path(f"{folder_path}/jpeg")

dir_raw = Path(f"{folder_path}/radar")
dir_camera = Path(f"{folder_path}/camera")
dir_lidar = Path(f"{folder_path}/lidar")

# background = np.load("/Benson_DATA3/Public/MUSE/data_70_80/background_content_70_80.npy")
# background = np.load(f"/Benson_DATA3/Public/MUSE/background_70_80.npy")
background = np.load("/home/skouff/background_db_with_plexi.npy")

background_puissance = np.load(f"/Benson_DATA3/Public/MUSE/background_puissance.npy")
save_path = Path("/home/skouff/master_thesis/kmd2_processing/")
SAVE_DIR = Path("/home/skouff/master_thesis/camera_calibration/calibration_files_paper")

# ==========================================
# LOAD FILES + TIMESTAMPS
# ==========================================
def load_files(folder, ext):
    files = sorted(folder.glob(f"*{ext}"))
    times = np.array([float(f.stem) for f in files])
    return files, times

raw_files, raw_times = load_files(dir_raw, ".raw")
cam_files, cam_times = load_files(dir_camera, ".jpeg")
lidar_files = sorted(dir_lidar.glob("*.ply"))
lidar_times = np.array([int(f.stem) / 1e9 for f in lidar_files])

def main(rd=True, camera=True, lidar=False):
    flags = {"rd": rd, "cam": camera, "lidar": lidar}
    panels = [name for name, active in flags.items() if active]
    viewer = RealTimeViewer(panels)
    writer = FFMpegWriter(fps=15)
    
    with writer.saving(viewer.fig, output_video, dpi=200):
        for i in range(1, len(raw_files)):
            print(f"Processing frame {i}/{len(raw_files)}")
            rd_power = compute_rd(raw_files[i], background=background_puissance, remove_background=True)

            cam_idx = find_closest_index(cam_times, raw_times[i])
            cam_src = cam_files[cam_idx]
            img = np.array(Image.open(cam_src))

            pts = None

            if lidar:
                lid_idx = find_closest_index(lidar_times, raw_times[i])
                pts = load_lidar(lidar_files[lid_idx])
            
            viewer.update(times=raw_times[i], camera_times=cam_times[i], rd_power=rd_power, img=img, lidar_time=lidar_times[i], pts=pts)
            if save_video:
                writer.grab_frame()

            plt.pause(0.001)
    
    # background_power = np.mean(backgrounds_power, axis=0)
    # np.save(f"/Benson_DATA3/Public/MUSE/background_puissance_hamming.npy", background_power)
    # print("/Benson_DATA3/Public/MUSE/background_puissance_hamming.npy")

    # background_content = np.mean(backgrounds_iq, axis=0)
    # np.save(f"{folder_path}/background_content_70_80", background_content)
    # print(background_content.shape)
    # print(f"Mean: {np.mean(backgrounds_db)}")
    
    plt.ioff()
    plt.show()
    
    print("Video saved:", output_video)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rd", action="store_true", default=True)
    parser.add_argument('--camera', action="store_true", default=True)
    parser.add_argument('--lidar', action="store_true", default=False)

    args = parser.parse_args()
    main(args.rd, args.camera, args.lidar)