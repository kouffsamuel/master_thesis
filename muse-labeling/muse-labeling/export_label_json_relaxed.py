import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN

from cfar import CA_CFAR
from radar_processing import compute_rd
from tracking import Tracking
from utils import find_closest_index, load_files


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


def pad_index(idx):
    return f"{idx:05d}"


def bin_to_point(r_bin, d_bin):
    r_idx = int(np.clip(round(float(r_bin)), 0, N - 1))
    d_idx = int(np.clip(round(float(d_bin)), 0, N - 1))
    return {
        "range_m": float(range_bins[r_idx]),
        "velocity_kmh": float(velocity_bins[d_idx]),
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


def yolo_boxes(model, image_path):
    img = np.array(Image.open(image_path))
    results = model.track(img, persist=True, tracker="bytetrack.yaml", verbose=False)
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for k in range(len(boxes)):
            detections.append(make_box_detection(boxes, model, k))
    return detections


def export(args):
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir
    raw_dir = dataset_dir / "radar"
    camera_dir = dataset_dir / "camera"
    rd_raw_dir = output_dir / "rd_raw"
    out_camera_dir = output_dir / "camera"

    output_dir.mkdir(parents=True, exist_ok=True)
    rd_raw_dir.mkdir(parents=True, exist_ok=True)
    out_camera_dir.mkdir(parents=True, exist_ok=True)

    raw_files, raw_times = load_files(raw_dir, ".raw")
    cam_files, cam_times = load_files(camera_dir, ".jpeg")
    if not raw_files:
        raise RuntimeError(f"No .raw files found in {raw_dir}")
    if not cam_files:
        raise RuntimeError(f"No .jpeg files found in {camera_dir}")

    background = np.load(args.background)
    if args.skip_yolo:
        model = None
    else:
        from ultralytics import YOLO

        model = YOLO(str(args.yolo_model))

    cfar = CA_CFAR(
        win_param=(15, 20, 9, 10),
        threshold=args.cfar_threshold,
        rd_size=(N, N),
    )
    dbscan = DBSCAN(eps=args.dbscan_eps, min_samples=args.dbscan_min_samples)
    tracker = Tracking()

    start = args.start
    end = args.end if args.end is not None else len(raw_files)
    end = min(end, len(raw_files))
    if start < 0 or start >= end:
        raise RuntimeError(f"Invalid frame range: start={start}, end={end}")

    data = {}
    for i in range(start, end):
        fi = pad_index(i)
        print(f"Exporting frame {i}/{end - 1}")

        rd_power = compute_rd(raw_files[i], background=background, remove_background=True)
        rd_power_wo = compute_rd(raw_files[i], background=background, remove_background=False)
        rd_raw_rel = f"rd_raw/frame_{fi}_rd.raw"
        (output_dir / rd_raw_rel).parent.mkdir(parents=True, exist_ok=True)
        rd_power.astype("<f4").tofile(output_dir / rd_raw_rel)

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
        track_by_id = {t.track_id: t for t in tracker.tracks}

        radar_detections = []
        for cluster in clusters:
            track_id = cluster.get("track_id", -1)
            track = track_by_id.get(track_id)
            r_bin, d_bin = cluster["centroid"]
            radar_detections.append({
                "track_id": int(track_id),
                "is_confirmed": bool(track.is_confirmed) if track is not None else False,
                "centroid": bin_to_point(r_bin, d_bin),
                "points": [
                    bin_to_point(r_bin=p[1], d_bin=p[0])
                    for p in cluster.get("points", [])
                ],
                "power": float(cluster["power"]),
            })

        track_history = []
        for track in tracker.get_confirmed_tracks():
            history = [
                bin_to_point(r_bin=centroid[0], d_bin=centroid[1])
                for centroid in track.centroid_history
            ]
            track_history.append({
                "track_id": int(track.track_id),
                "history": history,
            })

        cam_idx = find_closest_index(cam_times, raw_times[i])
        cam_src = cam_files[cam_idx]
        cam_rel = f"camera/frame_{fi}_camera.jpeg"
        shutil.copy2(cam_src, output_dir / cam_rel)

        data[f"frame_{fi}.jpeg"] = {
            "frame_index": i,
            "source_radar_file": raw_files[i].name,
            "source_camera_file": cam_src.name,
            "rd_raw_file": rd_raw_rel,
            "box_detections": [] if model is None else yolo_boxes(model, cam_src),
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

    json_path = output_dir / "yolo_tracking_data.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {json_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export relaxed radar clusters/tracks for the MUSE labeling tool."
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("./2026_05_20_13_30_38"))
    parser.add_argument("--background", type=Path, default=Path("./background_puissance.npy"))
    parser.add_argument("--yolo-model", type=Path, default=Path("./yolo26n.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("./label_export_relaxed_cfar9_dbscan2"))
    parser.add_argument("--start", type=int, default=100)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--cfar-threshold", type=float, default=9.0)
    parser.add_argument("--dbscan-eps", type=float, default=2.0)
    parser.add_argument("--dbscan-min-samples", type=int, default=2)
    parser.add_argument("--peak-metric", choices=["mean", "max", "median"], default="mean")
    parser.add_argument("--skip-yolo", action="store_true", help="Leave camera boxes empty to export radar data faster.")
    return parser.parse_args()


if __name__ == "__main__":
    export(parse_args())
