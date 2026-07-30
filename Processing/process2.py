# '''

# python process2.py \
#   --start 100 \
#   --end 1187 \
#   --frames-dir ../my_output_frames6 \
#   --no-video

# '''

# from collections import defaultdict
# import argparse
# import os
# import shutil
# import numpy as np
# import multiprocessing as mp
# import json
# from pathlib import Path
# from PIL import Image

# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt

# from cfar import CA_CFAR
# from tracking import Tracking
# from sklearn.cluster import DBSCAN
# from ultralytics import YOLO
# from utils import load_files, find_closest_index
# from radar_processing import compute_rd

# # ==========================================
# # PATHS & CONFIG
# # ==========================================
# PROJECT_ROOT = Path(__file__).resolve().parents[1]

# # Default input directory (override with --input-dir)
# input_dir_default = PROJECT_ROOT / "2026_05_20_13_30_38"

# output_video_default = PROJECT_ROOT / "video.mp4"
# output_frames_default = PROJECT_ROOT / "my_output_frames"

# background = np.load(PROJECT_ROOT / "background_db.npy")
# background_puissance = np.load(PROJECT_ROOT / "background_puissance.npy")
# save_path = PROJECT_ROOT / "MUSE"

# c = 3e8
# fc = 24.125e9
# lam = c / fc
# BW = 554e6
# N = 256
# clk = 38461538
# delay = 2214

# delta_v = (lam * clk * 3.6) / (2 * N * (12 * (N + 4) + delay))
# Vmax = delta_v * (N // 2)

# range_bins = np.arange(N) * (c / (2 * BW))
# velocity_bins = np.arange(N) * delta_v - Vmax
# p_noise = 10 ** (82.03/10)

# # Loaded later in setup_input() based on --input-dir
# folder_path = None
# dir_raw = None
# dir_camera = None
# raw_files, raw_times = [], []
# cam_files, cam_times = [], []
# camera_cache = {}

# model = YOLO("yolo26n.pt")
# CMAP = plt.get_cmap("tab20")


# def setup_input(input_dir):
#     """Load radar / camera files and cache from the given input directory."""
#     global folder_path, dir_raw, dir_camera
#     global raw_files, raw_times, cam_files, cam_times, camera_cache

#     folder_path = Path(input_dir)
#     dir_raw = folder_path / "radar"
#     dir_camera = folder_path / "camera"

#     raw_files, raw_times = load_files(dir_raw, ".raw")
#     cam_files, cam_times = load_files(dir_camera, ".jpeg")
#     camera_cache = {f: np.array(Image.open(f)) for f in cam_files}

#     print(f"Loaded input directory: {folder_path}")
#     print(f"  radar files: {len(raw_files)}, camera files: {len(cam_files)}")


# # ==========================================
# # 把 radar 偵測(cluster + tracking)轉成 JSON
# # 對應前端第二張圖 (radar_detections)
# # ==========================================
# def radar_detections_to_json(clusters, confirmed_track_ids, range_bins, velocity_bins, N):
#     """
#     把 DBSCAN cluster + tracking 結果轉成 JSON 格式。
#     每個偵測包含：
#     - track_id      : 這一幀內的 candidate key，同時是跨幀的持續身份
#     - is_confirmed  : 是否為確認的 track (hits >= 3)
#     - centroid      : 代表點 (range_m, velocity_kmh)
#     - points        : 該 cluster 所有原始點
#     注意：標注結果(object/noise/pair)不存這裡，改存在 frame 層級的 labeling。
#     """
#     radar_cluster_list = []

#     for c_obj in clusters:
#         track_id = int(c_obj.get("track_id", -1))

#         # --- centroid ---
#         r_bin, d_bin = c_obj["centroid"]
#         r_bin = int(np.clip(round(r_bin), 0, N - 1))
#         d_bin = int(np.clip(round(d_bin), 0, N - 1))

#         centroid = {
#             "range_m": float(range_bins[r_bin]),
#             "velocity_kmh": float(velocity_bins[d_bin])
#         }

#         # --- all points ---
#         points_list = []
#         if "points" in c_obj:
#             for (pb, rb) in c_obj["points"]:
#                 pb = int(np.clip(round(pb), 0, N - 1))
#                 rb = int(np.clip(round(rb), 0, N - 1))
#                 points_list.append({
#                     "range_m": float(range_bins[rb]),
#                     "velocity_kmh": float(velocity_bins[pb])
#                 })

#         radar_cluster_list.append({
#             "track_id": track_id,
#             "is_confirmed": track_id in confirmed_track_ids,
#             "centroid": centroid,
#             # Mean cluster power from extract_clusters (peak_met="mean").
#             # Computed on rd_power_wo (background NOT removed) — linear scale,
#             # so values are large; frontend displays it as 10*log10 dB.
#             "energy": float(c_obj.get("power", 0.0)),
#             "points": points_list
#         })

#     return radar_cluster_list


# # ==========================================
# # 把 track history 轉成 JSON 可序列化格式
# # ==========================================
# def track_histories_to_json(confirmed_tracks, range_bins, velocity_bins, N):
#     """
#     把所有 confirmed track 的完整移動歷史轉成 JSON 格式。
#     前端用這個畫第三張圖（Track History）。
#     每個 track 包含：
#     - track_id（決定顏色，前端用 track_id % 20 對應 tab20 colormap）
#     - history（每個時間點的 range_m 和 velocity_kmh）
#     """
#     histories = []

#     for tr in confirmed_tracks:
#         history = []
#         for r_bin, d_bin in tr.centroid_history:
#             r_bin = int(np.clip(round(r_bin), 0, N - 1))
#             d_bin = int(np.clip(round(d_bin), 0, N - 1))
#             history.append({
#                 "range_m": float(range_bins[r_bin]),
#                 "velocity_kmh": float(velocity_bins[d_bin])
#             })

#         histories.append({
#             "track_id": int(tr.track_id),
#             "history": history
#         })

#     return histories


# # ==========================================
# # 背景畫圖工人 (Consumer)
# # ==========================================
# def render_frame_worker(data):
#     (i, rd_power, peaks, clusters, simple_tracks, bboxes_data, img,
#      velocity_bins_worker, range_bins_worker, N_worker, output_dirs) = data

#     fig, (ax_rd, ax_cluster, ax_track, ax_cam) = plt.subplots(1, 4, figsize=(24, 6))

#     # --- RD 圖 ---
#     ax_rd.set_xlim(velocity_bins_worker[0], velocity_bins_worker[-1])
#     ax_rd.set_ylim(range_bins_worker[0], range_bins_worker[-1])
#     ax_rd.set_xlabel("Velocity (km/h)")
#     ax_rd.set_ylabel("Range (m)")
#     ax_rd.set_title(f"RD frame {i}")
#     ax_rd.imshow(
#         10 * np.log10(rd_power).T,
#         extent=[velocity_bins_worker[0], velocity_bins_worker[-1],
#                 range_bins_worker[0], range_bins_worker[-1]],
#         origin="lower", cmap="gray_r", aspect="auto", vmin=0, vmax=30
#     )

#     # --- Cluster 圖 ---
#     ax_cluster.set_xlim(velocity_bins_worker[0], velocity_bins_worker[-1])
#     ax_cluster.set_ylim(range_bins_worker[0], range_bins_worker[-1])
#     ax_cluster.set_xlabel("Velocity (km/h)")
#     ax_cluster.set_ylabel("Range (m)")
#     ax_cluster.set_title(f"Clusters frame - DBSCAN {i}")
#     ax_cluster.imshow(
#         peaks.T,
#         extent=[velocity_bins_worker[0], velocity_bins_worker[-1],
#                 range_bins_worker[0], range_bins_worker[-1]],
#         origin="lower", cmap="gray_r", aspect="auto", vmin=0, vmax=1
#     )

#     for c_obj in clusters:
#         track_id = c_obj.get("track_id", -1)
#         color = CMAP(track_id % CMAP.N) if track_id >= 0 else "red"
#         if "points" in c_obj:
#             xs, ys = [], []
#             for (d_bin, r_bin) in c_obj["points"]:
#                 d_bin = int(np.clip(round(d_bin), 0, N_worker - 1))
#                 r_bin = int(np.clip(round(r_bin), 0, N_worker - 1))
#                 xs.append(velocity_bins_worker[d_bin])
#                 ys.append(range_bins_worker[r_bin])
#             ax_cluster.scatter(xs, ys, c=[color], s=5, marker="s", alpha=0.6)
#         r_bin, d_bin = c_obj["centroid"]
#         d_bin = int(np.clip(round(d_bin), 0, N_worker - 1))
#         r_bin = int(np.clip(round(r_bin), 0, N_worker - 1))
#         ax_cluster.scatter(
#             velocity_bins_worker[d_bin], range_bins_worker[r_bin],
#             c=[color], s=80, marker="x", linewidths=2
#         )

#     # --- Camera 圖 ---
#     ax_cam.imshow(img)
#     ax_cam.axis("off")
#     ax_cam.set_title("Camera image")
#     for (x1, y1, x2, y2, label, conf, track_id) in bboxes_data:
#         rect = plt.Rectangle(
#             (x1, y1), x2 - x1, y2 - y1,
#             fill=False, linewidth=2, edgecolor="lime", zorder=10
#         )
#         ax_cam.add_patch(rect)
#         ax_cam.text(
#             x1, y1 - 5, f"{label} ID:{track_id} {conf:.2f}",
#             color="white",
#             bbox=dict(facecolor="black", alpha=0.6, edgecolor="none"),
#             zorder=10
#         )

#     # --- Track 圖 ---
#     ax_track.set_xlim(velocity_bins_worker[0], velocity_bins_worker[-1])
#     ax_track.set_ylim(range_bins_worker[0], range_bins_worker[-1])
#     ax_track.set_xlabel("Velocity (km/h)")
#     ax_track.set_ylabel("Range (m)")
#     ax_track.set_title("Range-Doppler Track History")
#     ax_track.grid(True, alpha=0.3)

#     for tr_id, centroid_history in simple_tracks:
#         color = CMAP(tr_id % CMAP.N)
#         ranges, velocities = [], []
#         for r_bin, d_bin in centroid_history:
#             r_bin = int(np.clip(round(r_bin), 0, N_worker - 1))
#             d_bin = int(np.clip(round(d_bin), 0, N_worker - 1))
#             ranges.append(range_bins_worker[r_bin])
#             velocities.append(velocity_bins_worker[d_bin])
#         if len(ranges) >= 2:
#             ax_track.plot(velocities, ranges, '-', color=color, linewidth=2)
#             ax_track.scatter(velocities[-1], ranges[-1], color=color, s=80)
#             ax_track.text(
#                 velocities[-1], ranges[-1], f"ID {tr_id}",
#                 color=color, fontsize=10, fontweight="bold"
#             )

#     # --- 存檔 ---
#     fig.canvas.draw()
#     renderer = fig.canvas.get_renderer()
#     fig.savefig(
#         Path(output_dirs["combined"]) / f"frame_{i:05d}.jpeg", dpi=fig.dpi
#     )

#     axes_map = {
#         "rd": ax_rd, "cluster": ax_cluster,
#         "track": ax_track, "camera": ax_cam
#     }
#     output_dir_map = {
#         "rd": "rd", "cluster": "cluster",
#         "track": "track", "camera": "camera_yolo"
#     }
#     for name, ax in axes_map.items():
#         extent = ax.get_tightbbox(renderer).transformed(
#             fig.dpi_scale_trans.inverted()
#         )
#         fig.savefig(
#             Path(output_dirs[output_dir_map[name]]) / f"frame_{i:05d}_{name}.jpeg",
#             bbox_inches=extent
#         )

#     plt.close(fig)


# # ==========================================
# # 主程式 (Producer)
# # ==========================================
# def tracking_and_clustering(
#     save_path, start_frame, end_frame, output_video, frames_dir, write_video=True
# ):
#     cfar_fonction = CA_CFAR(win_param=(15, 20, 9, 10), threshold=12, rd_size=(N, N))
#     dbscan = DBSCAN(eps=2, min_samples=3)
#     tracks = Tracking()
#     end_frame = min(end_frame, len(raw_files))

#     frames_dir.mkdir(parents=True, exist_ok=True)
#     sub_dirs = {
#         "combined": str(frames_dir / "combined"),
#         "rd": str(frames_dir / "rd"),
#         "rd_raw": str(frames_dir / "rd_raw"),   # rd_power 原始矩陣 (.raw)，供前端渲染
#         "cluster": str(frames_dir / "cluster"),
#         "track": str(frames_dir / "track"),
#         "camera": str(frames_dir / "camera"),
#         "camera_yolo": str(frames_dir / "camera_yolo")
#     }
#     for d in sub_dirs.values():
#         Path(d).mkdir(parents=True, exist_ok=True)

#     all_frames_json_data = {}

#     num_workers = max(1, mp.cpu_count() - 2)
#     pool = mp.Pool(processes=num_workers)
#     print(f"啟動平行管線處理... 使用 {num_workers} 個 CPU 核心進行背景繪圖")

#     for i in range(start_frame, end_frame):
#         print(f"Processing frame {i}/{len(raw_files)}")

#         rd_power = compute_rd(
#             raw_files[i], background=background_puissance, remove_background=True
#         )
#         rd_power_wo = compute_rd(
#             raw_files[i], background=background_puissance, remove_background=False
#         )

#         # ----- 存 rd_power 原始矩陣成 .raw，供前端自行渲染 RD map -----
#         # float32 + C-order，固定 256x256。前端用 Float32Array 讀進來後，
#         # 複製 render_frame_worker 的視覺化步驟：
#         #   10*log10(rd_power) -> 轉置(.T) -> clip(vmin=0, vmax=30) -> 灰階反轉(gray_r)
#         # 存的是 remove_background=True 的版本，跟畫 RD map jpeg 用的同一份。
#         rd_raw_name = f"frame_{i:05d}_rd.raw"
#         rd_power.astype(np.float32).tofile(
#             Path(sub_dirs["rd_raw"]) / rd_raw_name
#         )

#         peaks = cfar_fonction(rd_power)
#         detected_bins = np.where(peaks > 0)
#         dbscan.fit(np.array(detected_bins).T)
#         labels = dbscan.labels_
#         clusters = tracks.extract_clusters(
#             detected_bins, labels, rd_power_wo, peak_met="mean"
#         )
#         tracks.step(clusters)

#         t = raw_times[i]
#         idx = find_closest_index(cam_times, t)
#         img = camera_cache[cam_files[idx]]

#         src_cam_path = cam_files[idx]
#         shutil.copy(
#             src_cam_path,
#             Path(sub_dirs["camera"]) / f"frame_{i:05d}_camera{src_cam_path.suffix}"
#         )

#         results = model.track(
#             img, persist=True, tracker="bytetrack.yaml", verbose=False, device=0
#         )

#         bboxes_data = []
#         json_box_list = []

#         if results and results[0].boxes is not None:
#             boxes = results[0].boxes
#             for k in range(len(boxes)):
#                 x1, y1, x2, y2 = boxes.xyxy[k].cpu().numpy()
#                 cls = int(boxes.cls[k].cpu().numpy())
#                 conf = float(boxes.conf[k].cpu().numpy())
#                 track_id = int(boxes.id[k].cpu().numpy()) if boxes.id is not None else -1
#                 label = model.names[cls]

#                 bboxes_data.append((x1, y1, x2, y2, label, conf, track_id))
#                 json_box_list.append({
#                     "pos1": [float(x1), float(y1)],
#                     "pos2": [float(x2), float(y2)],
#                     "label": label,
#                     "confidence": float(conf),
#                     "track_id": int(track_id),
#                     "thickness": 2,
#                     # 原始 YOLO 輸出快照：跑完即定死。前端可編輯 pos1/pos2/label，
#                     # 之後仍能從這裡還原回 process.py 當次的偵測結果。
#                     "original": {
#                         "pos1": [float(x1), float(y1)],
#                         "pos2": [float(x2), float(y2)],
#                         "label": label,
#                         "confidence": float(conf)
#                     }
#                 })

#         # ==========================================
#         # 把 radar cluster 和 track history 寫進 JSON
#         # ==========================================
#         confirmed_tracks = tracks.get_confirmed_tracks()
#         confirmed_track_ids = {tr.track_id for tr in confirmed_tracks}

#         radar_detection_list = radar_detections_to_json(
#             clusters, confirmed_track_ids, range_bins, velocity_bins, N
#         )
#         track_history_list = track_histories_to_json(
#             confirmed_tracks, range_bins, velocity_bins, N
#         )

#         frame_filename = f"frame_{i:05d}.jpeg"
#         all_frames_json_data[frame_filename] = {
#             "frame_index": i,                        # 對應 raw/camera 檔案的 index
#             "source_camera_file": src_cam_path.name,
#             "rd_raw_file": f"rd_raw/{rd_raw_name}",   # RD map 原始矩陣路徑（第一張圖，前端渲染）
#             "box_detections": json_box_list,         # YOLO 框（第四張圖）
#             "radar_detections": radar_detection_list,# 當幀 radar 偵測（第二張圖）
#             "track_history": track_history_list,     # 完整軌跡歷史（第三張圖）

#             # ----- 標注結果：每一幀獨立的單一事實來源 -----
#             # 由 labeling-tool 前端填寫，process.py 只負責產生空殼。
#             #   object : { radar_track_id(str) : box_track_id 或 null }
#             #            radar track 是真實物體，value 是對應到的 YOLO 框 id；
#             #            若這一幀 camera 沒框到，value 為 null。
#             #   noise  : [ radar_track_id, ... ]  雜訊點
#             #   pairs  : [ [radar_id_a, radar_id_b], ... ]  鏡像配對(對稱關係只存一筆)
#             "labeling": {
#                 "object": {},
#                 "noise": [],
#                 "pairs": []
#             }
#         }

#         simple_tracks = [
#             (tr.track_id, list(tr.centroid_history)) for tr in confirmed_tracks
#         ]

#         worker_data = (
#             i, rd_power, peaks, clusters, simple_tracks, bboxes_data, img,
#             velocity_bins, range_bins, N, sub_dirs
#         )
#         pool.apply_async(render_frame_worker, args=(worker_data,))

#     print("主運算已完成，等待背景渲染圖片...")
#     pool.close()
#     pool.join()

#     json_output_path = frames_dir / "yolo_tracking_data.json"
#     with open(json_output_path, "w", encoding="utf-8") as f:
#         json.dump(all_frames_json_data, f, indent=4, ensure_ascii=False)

#     print(f"處理完畢！圖片與座標資料 (JSON) 已儲存至 {frames_dir}。")


# # ==========================================
# # MAIN
# # ==========================================
# def main():
#     parser = argparse.ArgumentParser(
#         description="Batch-render MUSE radar/camera processing output."
#     )
#     parser.add_argument("--input-dir", default=str(input_dir_default),
#                         help="Input folder containing radar/ and camera/ subdirectories.")
#     parser.add_argument("--start", type=int, default=100)
#     parser.add_argument("--end", type=int, default=600)
#     parser.add_argument("--output-video", default=str(output_video_default))
#     parser.add_argument("--frames-dir", default=str(output_frames_default))
#     parser.add_argument("--no-video", action="store_true")
#     args = parser.parse_args()

#     mp.freeze_support()

#     # Load radar/camera files from the chosen input directory
#     setup_input(args.input_dir)

#     try:
#         tracking_and_clustering(
#             save_path,
#             args.start,
#             args.end,
#             Path(args.output_video),
#             Path(args.frames_dir),
#             write_video=not args.no_video,
#         )
#     except KeyboardInterrupt:
#         print("\nInterrupted by user")
#         plt.close("all")


# if __name__ == "__main__":
#     main()

"""
Unified MUSE processing/export pipeline.

Examples:
  python Processing/process.py --start 100 --end 1187 --frames-dir ./my_output_frames5 --no-video

  python Processing/process.py --export-only --output-dir ./label_export \
    --cfar-threshold 9.0 --dbscan-min-samples 2 --peak-metric mean --skip-yolo

            python process2.py --export-only \
                --start 100 --end 600 \
                --cfar-threshold 12.0 --dbscan-eps 2.0 --dbscan-min-samples 3 \
                --peak-metric mean \
                --dataset-dir ../DATA/day-1 \
                --output-dir day1_cfar12_ms3
"""

import argparse
import json
import multiprocessing as mp
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN

from cfar import CA_CFAR
from radar_processing import compute_rd
from tracking import Tracking
from utils import find_closest_index, load_files


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASET_DIR = PROJECT_ROOT / "DATA" / "Day-1"
DEFAULT_FRAMES_DIR = PROJECT_ROOT / "my_output_frames"
DEFAULT_EXPORT_DIR = PROJECT_ROOT / "label_export"
DEFAULT_BACKGROUND = PROJECT_ROOT / "background_puissance.npy"
DEFAULT_YOLO_MODEL = PROJECT_ROOT / "yolo26n.pt"

c = 3e8
fc = 24.125e9
lam = c / fc
BW = 554e6
N = 256
clk = 38461538
delay = 2214

delta_v = (lam * clk * 3.6) / (2 * N * (12 * (N + 4) + delay))
Vmax = delta_v * (N // 2)
range_bins = np.arange(N) * (c / (2 * BW))
velocity_bins = np.arange(N) * delta_v - Vmax

CMAP = plt.get_cmap("tab20")


def pad_index(idx):
    return f"{idx:05d}"


def bin_to_point(r_bin, d_bin):
    r_idx = int(np.clip(round(float(r_bin)), 0, N - 1))
    d_idx = int(np.clip(round(float(d_bin)), 0, N - 1))
    # Round to 4 decimals (0.1 mm / sub-km/h precision) — full float64
    # repr wastes roughly half the JSON size for precision nobody consumes.
    return {
        "range_m": round(float(range_bins[r_idx]), 4),
        "velocity_kmh": round(float(velocity_bins[d_idx]), 4),
    }


def make_box_detection(boxes, model, k):
    x1, y1, x2, y2 = boxes.xyxy[k].cpu().numpy()
    cls = int(boxes.cls[k].cpu().numpy())
    conf = float(boxes.conf[k].cpu().numpy())
    track_id = None
    if boxes.id is not None:
        track_id = int(boxes.id[k].cpu().numpy())

    det = {
        "pos1": [float(x1), float(y1)],
        "pos2": [float(x2), float(y2)],
        "label": model.names[cls],
        "confidence": conf,
        "track_id": track_id,
        "thickness": 2,
    }
    det["original"] = {
        "pos1": det["pos1"][:],
        "pos2": det["pos2"][:],
        "label": det["label"],
        "confidence": det["confidence"],
    }
    return det


def yolo_boxes(model, img, device=None):
    kwargs = {
        "persist": True,
        "tracker": "bytetrack.yaml",
        "verbose": False,
    }
    if device is not None:
        kwargs["device"] = device

    results = model.track(img, **kwargs)
    detections = []
    bboxes_for_render = []

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for k in range(len(boxes)):
            det = make_box_detection(boxes, model, k)
            detections.append(det)
            bboxes_for_render.append((
                det["pos1"][0],
                det["pos1"][1],
                det["pos2"][0],
                det["pos2"][1],
                det["label"],
                det["confidence"],
                det["track_id"],
            ))

    return detections, bboxes_for_render


def radar_detections_to_json(clusters, track_by_id):
    radar_detection_list = []

    for cluster in clusters:
        track_id = int(cluster.get("track_id", -1))
        track = track_by_id.get(track_id)
        r_bin, d_bin = cluster["centroid"]
        radar_detection_list.append({
            "track_id": track_id,
            "is_confirmed": bool(track.is_confirmed) if track is not None else False,
            "centroid": bin_to_point(r_bin, d_bin),
            "points": [
                bin_to_point(r_bin=p[1], d_bin=p[0])
                for p in cluster.get("points", [])
            ],
            # Mean cluster power from extract_clusters (peak_met).
            # Computed on rd_power_wo (background NOT removed) — linear scale;
            # the frontend displays it as 10*log10 dB and Stage 1 thresholds on it.
            # Field name "energy" is the contract with the labeling tool.
            "energy": float(cluster["power"]),
        })

    return radar_detection_list


HISTORY_CAP = 60  # per-frame history entries kept per track


def track_histories_to_json(confirmed_tracks):
    histories = []

    for track in confirmed_tracks:
        # Cap the exported history to the most recent HISTORY_CAP entries.
        # Every frame re-serializes every track's FULL history, so total
        # JSON size grows O(frames^2) for long-lived tracks (the fixed
        # noise lives for the whole recording) — this is what pushed the
        # file past the browser's ~512MB string limit on complex scenes.
        # 60 covers MAX_MISSES (40) with margin; Stage 1's footprint
        # check becomes "left the region within the last 60 frames",
        # an accepted narrowing (energy check + human review back it up).
        recent = track.centroid_history[-HISTORY_CAP:]
        histories.append({
            "track_id": int(track.track_id),
            "history": [
                bin_to_point(r_bin=centroid[0], d_bin=centroid[1])
                for centroid in recent
            ],
        })

    return histories


def render_frame_worker(data):
    (
        i,
        rd_power,
        peaks,
        clusters,
        simple_tracks,
        bboxes_data,
        img,
        output_dirs,
    ) = data

    fig, (ax_rd, ax_cluster, ax_track, ax_cam) = plt.subplots(1, 4, figsize=(24, 6))

    ax_rd.set_xlim(velocity_bins[0], velocity_bins[-1])
    ax_rd.set_ylim(range_bins[0], range_bins[-1])
    ax_rd.set_xlabel("Velocity (km/h)")
    ax_rd.set_ylabel("Range (m)")
    ax_rd.set_title(f"RD frame {i}")
    ax_rd.imshow(
        10 * np.log10(rd_power).T,
        extent=[velocity_bins[0], velocity_bins[-1], range_bins[0], range_bins[-1]],
        origin="lower",
        cmap="gray_r",
        aspect="auto",
        vmin=0,
        vmax=30,
    )

    ax_cluster.set_xlim(velocity_bins[0], velocity_bins[-1])
    ax_cluster.set_ylim(range_bins[0], range_bins[-1])
    ax_cluster.set_xlabel("Velocity (km/h)")
    ax_cluster.set_ylabel("Range (m)")
    ax_cluster.set_title(f"Clusters frame - DBSCAN {i}")
    ax_cluster.imshow(
        peaks.T,
        extent=[velocity_bins[0], velocity_bins[-1], range_bins[0], range_bins[-1]],
        origin="lower",
        cmap="gray_r",
        aspect="auto",
        vmin=0,
        vmax=1,
    )

    for cluster in clusters:
        track_id = int(cluster.get("track_id", -1))
        color = CMAP(track_id % CMAP.N) if track_id >= 0 else "red"
        xs, ys = [], []
        for d_bin, r_bin in cluster.get("points", []):
            d_bin = int(np.clip(round(d_bin), 0, N - 1))
            r_bin = int(np.clip(round(r_bin), 0, N - 1))
            xs.append(velocity_bins[d_bin])
            ys.append(range_bins[r_bin])
        ax_cluster.scatter(xs, ys, c=[color], s=5, marker="s", alpha=0.6)

        r_bin, d_bin = cluster["centroid"]
        d_bin = int(np.clip(round(d_bin), 0, N - 1))
        r_bin = int(np.clip(round(r_bin), 0, N - 1))
        ax_cluster.scatter(
            velocity_bins[d_bin],
            range_bins[r_bin],
            c=[color],
            s=80,
            marker="x",
            linewidths=2,
        )

    ax_cam.imshow(img)
    ax_cam.axis("off")
    ax_cam.set_title("Camera image")
    for x1, y1, x2, y2, label, conf, track_id in bboxes_data:
        rect = plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            linewidth=2,
            edgecolor="lime",
            zorder=10,
        )
        ax_cam.add_patch(rect)
        ax_cam.text(
            x1,
            y1 - 5,
            f"{label} ID:{track_id} {conf:.2f}",
            color="white",
            bbox=dict(facecolor="black", alpha=0.6, edgecolor="none"),
            zorder=10,
        )

    ax_track.set_xlim(velocity_bins[0], velocity_bins[-1])
    ax_track.set_ylim(range_bins[0], range_bins[-1])
    ax_track.set_xlabel("Velocity (km/h)")
    ax_track.set_ylabel("Range (m)")
    ax_track.set_title("Range-Doppler Track History")
    ax_track.grid(True, alpha=0.3)

    for tr_id, centroid_history in simple_tracks:
        color = CMAP(tr_id % CMAP.N)
        ranges, velocities = [], []
        for r_bin, d_bin in centroid_history:
            r_bin = int(np.clip(round(r_bin), 0, N - 1))
            d_bin = int(np.clip(round(d_bin), 0, N - 1))
            ranges.append(range_bins[r_bin])
            velocities.append(velocity_bins[d_bin])
        if len(ranges) >= 2:
            ax_track.plot(velocities, ranges, "-", color=color, linewidth=2)
            ax_track.scatter(velocities[-1], ranges[-1], color=color, s=80)
            ax_track.text(
                velocities[-1],
                ranges[-1],
                f"ID {tr_id}",
                color=color,
                fontsize=10,
                fontweight="bold",
            )

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig.savefig(Path(output_dirs["combined"]) / f"frame_{i:05d}.jpeg", dpi=fig.dpi)

    axes_map = {
        "rd": ax_rd,
        "cluster": ax_cluster,
        "track": ax_track,
        "camera": ax_cam,
    }
    output_dir_map = {
        "rd": "rd",
        "cluster": "cluster",
        "track": "track",
        "camera": "camera_yolo",
    }
    for name, ax in axes_map.items():
        extent = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(
            Path(output_dirs[output_dir_map[name]]) / f"frame_{i:05d}_{name}.jpeg",
            bbox_inches=extent,
        )

    plt.close(fig)


def prepare_output_dirs(output_dir, render_frames):
    sub_dirs = {
        "rd_raw": output_dir / "rd_raw",
        "camera": output_dir / "camera",
    }
    if render_frames:
        sub_dirs.update({
            "combined": output_dir / "combined",
            "rd": output_dir / "rd",
            "cluster": output_dir / "cluster",
            "track": output_dir / "track",
            "camera_yolo": output_dir / "camera_yolo",
        })

    for path in sub_dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return {key: str(path) for key, path in sub_dirs.items()}


def load_dataset(dataset_dir):
    raw_dir = dataset_dir / "radar"
    camera_dir = dataset_dir / "camera"
    raw_files, raw_times = load_files(raw_dir, ".raw")
    cam_files, cam_times = load_files(camera_dir, ".jpeg")

    if not raw_files:
        raise RuntimeError(f"No .raw files found in {raw_dir}")
    if not cam_files:
        raise RuntimeError(f"No .jpeg files found in {camera_dir}")

    print(f"Loaded input directory: {dataset_dir}")
    print(f"  radar files: {len(raw_files)}, camera files: {len(cam_files)}")
    return raw_files, raw_times, cam_files, cam_times


def run_pipeline(args):
    raw_files, raw_times, cam_files, cam_times = load_dataset(args.dataset_dir)

    background = np.load(args.background)
    cfar = CA_CFAR(
        win_param=(15, 20, 9, 10),
        threshold=args.cfar_threshold,
        rd_size=(N, N),
    )
    dbscan = DBSCAN(eps=args.dbscan_eps, min_samples=args.dbscan_min_samples)
    tracker = Tracking()

    model = None
    if not args.skip_yolo:
        from ultralytics import YOLO

        model = YOLO(str(args.yolo_model))

    output_dir = args.output_dir if args.output_dir is not None else args.frames_dir
    output_dirs = prepare_output_dirs(output_dir, render_frames=not args.export_only)

    start = args.start
    end = args.end if args.end is not None else len(raw_files)
    end = min(end, len(raw_files))
    if start < 0 or start >= end:
        raise RuntimeError(f"Invalid frame range: start={start}, end={end}")

    pool = None
    if not args.export_only:
        num_workers = max(1, mp.cpu_count() - 2)
        pool = mp.Pool(processes=num_workers)
        print(f"Using {num_workers} CPU workers for background rendering")

    all_frames_json_data = {}

    for i in range(start, end):
        fi = pad_index(i)
        print(f"Processing frame {i}/{end - 1}")

        rd_power = compute_rd(raw_files[i], background=background, remove_background=True)
        rd_power_wo = compute_rd(raw_files[i], background=background, remove_background=False)

        rd_raw_name = f"frame_{fi}_rd.raw"
        rd_raw_path = Path(output_dirs["rd_raw"]) / rd_raw_name
        rd_power.astype("<f4").tofile(rd_raw_path)

        peaks = cfar(rd_power)
        detected_bins = np.where(peaks > 0)
        if len(detected_bins[0]) > 0:
            dbscan.fit(np.array(detected_bins).T)
            labels = dbscan.labels_
            clusters = tracker.extract_clusters(
                detected_bins,
                labels,
                rd_power_wo,
                peak_met=args.peak_metric,
            )
        else:
            clusters = []

        tracker.step(clusters)
        track_by_id = {track.track_id: track for track in tracker.tracks}

        cam_idx = find_closest_index(cam_times, raw_times[i])
        cam_src = cam_files[cam_idx]
        img = np.array(Image.open(cam_src))
        cam_rel = f"camera/frame_{fi}_camera{cam_src.suffix}"
        shutil.copy2(cam_src, output_dir / cam_rel)

        if model is None:
            box_detections, bboxes_for_render = [], []
        else:
            box_detections, bboxes_for_render = yolo_boxes(model, img, device=args.device)

        confirmed_tracks = tracker.get_confirmed_tracks()
        radar_detections = radar_detections_to_json(clusters, track_by_id)
        track_history = track_histories_to_json(confirmed_tracks)

        frame_filename = f"frame_{fi}.jpeg"
        all_frames_json_data[frame_filename] = {
            "frame_index": i,
            "source_radar_file": raw_files[i].name,
            "source_camera_file": cam_src.name,
            "rd_raw_file": f"rd_raw/{rd_raw_name}",
            "box_detections": box_detections,
            "radar_detections": radar_detections,
            "track_history": track_history,
            "labeling": {
                "object": {},
                "noise": [],
                "pairs": [],
            },
            "metadata": {
                "cfar_threshold": args.cfar_threshold,
                "dbscan_eps": args.dbscan_eps,
                "dbscan_min_samples": args.dbscan_min_samples,
                "peak_metric": args.peak_metric,
            },
        }

        if pool is not None:
            simple_tracks = [
                (track.track_id, list(track.centroid_history))
                for track in confirmed_tracks
            ]
            worker_data = (
                i,
                rd_power,
                peaks,
                clusters,
                simple_tracks,
                bboxes_for_render,
                img,
                output_dirs,
            )
            pool.apply_async(render_frame_worker, args=(worker_data,))

    if pool is not None:
        print("Main processing finished; waiting for rendered images...")
        pool.close()
        pool.join()

    json_output_path = output_dir / "yolo_tracking_data.json"
    with json_output_path.open("w", encoding="utf-8") as f:
        # Compact output (no indent, no spaces): pretty-printing roughly
        # doubles file size at this nesting depth and nothing human-reads
        # the raw file — the labeling tool is the reader.
        json.dump(all_frames_json_data, f, separators=(",", ":"), ensure_ascii=False)

    print(f"Wrote {json_output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process MUSE radar/camera data and export labeling JSON."
    )
    parser.add_argument(
        "--dataset-dir",
        "--input-dir",
        dest="dataset_dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Input folder containing radar/ and camera/ subdirectories.",
    )
    parser.add_argument("--background", type=Path, default=DEFAULT_BACKGROUND)
    parser.add_argument("--yolo-model", type=Path, default=DEFAULT_YOLO_MODEL)
    parser.add_argument("--frames-dir", type=Path, default=DEFAULT_FRAMES_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    # Defaults match the original hardcoded values in process2.py
    # (CA_CFAR threshold=12, DBSCAN eps=2, min_samples=3, peak_met="mean"),
    # so running with no arguments reproduces the current authoritative snapshot.
    parser.add_argument("--cfar-threshold", type=float, default=12.0)
    parser.add_argument("--dbscan-eps", type=float, default=2.0)
    parser.add_argument("--dbscan-min-samples", type=int, default=3)
    parser.add_argument("--peak-metric", choices=["mean", "max", "median"], default="mean")
    parser.add_argument("--skip-yolo", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--device", default=None, help="Optional YOLO device, for example 0 or cpu.")
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Compatibility flag from process2.py; this script renders frames, not video.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output_dir is None and args.export_only:
        args.output_dir = DEFAULT_EXPORT_DIR

    mp.freeze_support()
    try:
        run_pipeline(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        plt.close("all")


if __name__ == "__main__":
    main()