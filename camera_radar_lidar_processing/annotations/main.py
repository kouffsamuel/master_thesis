import argparse

import numpy as np
from annotations.camera_master import CameraMaster
from annotations.frame_data import FrameProcessor
from annotations.detection_recorder import DetectionRecorder
from annotations.real_time_viewer import RealTimeViewer
from matplotlib.animation import FFMpegWriter
from pathlib import Path
import os
import matplotlib.pyplot as plt

def load_files(folder, ext):
    files = sorted(folder.glob(f"*{ext}"))
    times = np.array([float(f.stem) for f in files])
    return files, times

def main(folder_path, background, output_video, output_json):
    dir_raw = Path(f"{folder_path}/radar")
    dir_camera = Path(f"{folder_path}/camera")
    dir_lidar = Path(f"{folder_path}/lidar")
    background = np.load(background)

    raw_files, raw_times = load_files(dir_raw, ".raw")
    raw_files = raw_files[1:]
    cam_files, cam_times = load_files(dir_camera, ".jpeg")
    lidar_files = sorted(dir_lidar.glob("*.ply"))
    lidar_times = np.array([int(f.stem) / 1e9 for f in lidar_files])
    
    calibration = np.load("/home/skouff/master_thesis/camera_calibration/calibration.npz")
    cam_matrix = calibration["camMatrix"]
    dist_coeff = calibration["distCoeff"]

    master = CameraMaster(
        yolo_model_path="/home/skouff/yolo26x.pt",
        cam_matrix=cam_matrix,
        dist_coeff=dist_coeff,
    )

    processor = FrameProcessor(master, raw_files, raw_times, cam_files, cam_times, lidar_files, lidar_times, background)
    recorder = DetectionRecorder(Path(folder_path).name)
    viewer = RealTimeViewer(cam_matrix, dist_coeff)

    writer = FFMpegWriter(fps=14)
    print(folder_path)
    print(os.path.isdir(folder_path))

    with writer.saving(viewer.fig, output_video, dpi=200):
        for i in range(20, 50):
            print(f"Frame: {i}/{len(raw_files)}")

            frame = processor.process(i)
            viewer.update(frame)
            recorder.record_frame(frame)

            
            writer.grab_frame()
            plt.pause(0.001)

    recorder.save(output_json)

    plt.ioff()
    plt.show()
    print("Video saved:", output_video)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Process MUSE radar/camera/lidar data and export labeling JSON."
        )
    parser.add_argument("--folder_path", type=str, help="Folder path of data", default="/DATA_MUSE/old/2026_08_07_10_05_00/")
    parser.add_argument("--background", type=str, help="Path of background radar", default="/home/skouff/background_power_with_plexi.npy")
    parser.add_argument("--output_video", type=str, help="Path of video", default="/home/skouff/video_test.mp4")
    parser.add_argument("--output_json", type=str, help="Path of json file", default="/home/skouff/master_thesis/camera_radar_lidar_processing/annotations/labels.json")
    args = parser.parse_args()
    main(args.folder_path, args.background, args.output_video, args.output_json)