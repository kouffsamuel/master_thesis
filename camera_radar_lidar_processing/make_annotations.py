"""
Unified MUSE processing/export pipeline.

Examples:
    python make_annotations.py 
    --input-dir "/Benson_DATA3/Public/MUSE/data_route_2-cam_trot" 
    --background "/Benson_DATA3/Public/MUSE/background_puissance_hamming.npy"
"""

import argparse
import json
import multiprocessing as mp
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN

from master_thesis.camera_radar_lidar_processing.processing.cfar import CA_CFAR
from master_thesis.camera_radar_lidar_processing.processing.radar_processing import compute_rd, clusterize_radar
from tracking import Tracking
from master_thesis.camera_radar_lidar_processing.processing.utils import find_closest_index, load_files
from real_time_viewer import RealTimeViewer
from master_thesis.camera_radar_lidar_processing.processing.radar_parameters import N, velocity_bins, range_bins
from scipy.ndimage import binary_closing, binary_opening


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASET_DIR = PROJECT_ROOT / "DATA" / "Day-1"
DEFAULT_FRAMES_DIR = PROJECT_ROOT / "my_output_frames"
DEFAULT_EXPORT_DIR = PROJECT_ROOT / "label_export"
DEFAULT_BACKGROUND = PROJECT_ROOT / "background_puissance.npy"
DEFAULT_YOLO_MODEL = PROJECT_ROOT / "yolo26n.pt"

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


def load_dataset(dataset_dir):
    raw_dir = Path(f"{dataset_dir}/raw")
    camera_dir = Path(f"{dataset_dir}/jpeg")
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
    viewer = RealTimeViewer(panels=("rd", "cluster", "track", "cam"))
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

    start = args.start
    end = args.end if args.end is not None else len(raw_files)
    end = min(end, len(raw_files))
    if start < 0 or start >= end:
        raise RuntimeError(f"Invalid frame range: start={start}, end={end}")

    all_frames_json_data = {}

    for i in range(start, end):
        fi = pad_index(i)
        print(f"Processing frame {i}/{end - 1}")

        rd_power = compute_rd(raw_files[i], background=background, remove_background=True)
        rd_power_wo = compute_rd(raw_files[i], background=background, remove_background=False)

        rd_raw_name = f"frame_{fi}_rd.raw"

        clusters, peaks = clusterize_radar(rd_power, rd_power_wo, cfar, dbscan, args.peak_metric)
        

        tracker.step(clusters)
        track_by_id = {track.track_id: track for track in tracker.tracks}

        cam_idx = find_closest_index(cam_times, raw_times[i])
        cam_src = cam_files[cam_idx]
        img = np.array(Image.open(cam_src))

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

        simple_tracks = [
                        (track.track_id, list(track.centroid_history))
                        for track in confirmed_tracks
                    ]  
        viewer.update(times=raw_times[i], 
                      camera_times=cam_times[i], 
                      rd_power=rd_power, 
                      img=img, 
                      peaks=peaks, 
                      clusters=clusters, 
                      bboxes_data=bboxes_for_render, 
                      simple_tracks=simple_tracks, 
                      frame_idx=i)
          


    json_output_path = "yolo_tracking_data.json"
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
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    # Defaults match the original hardcoded values in process2.py
    # (CA_CFAR threshold=12, DBSCAN eps=2, min_samples=3, peak_met="mean"),
    # so running with no arguments reproduces the current authoritative snapshot.
    parser.add_argument("--cfar-threshold", type=float, default=10.0)
    parser.add_argument("--dbscan-eps", type=float, default=10.0)
    parser.add_argument("--dbscan-min-samples", type=int, default=5)
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

    mp.freeze_support()
    try:
        run_pipeline(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        plt.close("all")


if __name__ == "__main__":
    main()