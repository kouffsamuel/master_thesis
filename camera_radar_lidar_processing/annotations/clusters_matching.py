import numpy as np
from annotations.lidar_projection import project_point_to_camera, lidar_to_radar_range

def match_clusters_to_detections(clusters, camera_detections, camMatrix, distCoeff,
                                  R=None, t=None, max_distance_px=None):

    det_centers = [det["center"] for det in camera_detections]
    det_ids = [det["id"] for det in camera_detections]

    for cluster in clusters:
        proj = project_point_to_camera(cluster["center"], camMatrix, distCoeff, R, t)

        if proj is None or len(det_centers) == 0:
            cluster["detection_id"] = None
            cluster["pixel_distance"] = None
            continue

        u, v = proj

        best_idx = None
        best_dist = None
        for didx, (dcx, dcy) in enumerate(det_centers):
            dist = np.hypot(u - dcx, v - dcy)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_idx = didx

        if max_distance_px is not None and best_dist > max_distance_px:
            cluster["detection_id"] = None
            cluster["pixel_distance"] = None
        else:
            cluster["detection_id"] = det_ids[best_idx]
            cluster["pixel_distance"] = best_dist

    return clusters

def match_radar_clusters_to_lidar(clusters_lidar, clusters_radar, range_bins, N, max_range_diff_m):
    radar_ranges = []
    for cluster in clusters_radar:
        r_bin, d_bin = cluster["centroid"]
        r_bin = int(np.clip(round(r_bin), 0, N - 1))
        radar_ranges.append(range_bins[r_bin])
    radar_ranges = np.array(radar_ranges)

    for cluster in clusters_lidar:
        det_id = cluster.get("detection_id")
        if det_id is None or len(radar_ranges) == 0:
            continue

        target_range = lidar_to_radar_range(cluster["center"])
        closest_idx = np.argmin(np.abs(radar_ranges - target_range))
        range_diff = abs(radar_ranges[closest_idx] - target_range)

        if max_range_diff_m is not None and range_diff > max_range_diff_m:
            clusters_radar[closest_idx]["detection_id"] = None
            continue
        
        clusters_radar[closest_idx]["detection_id"] = det_id
        clusters_radar[closest_idx]["range_distance_to_lidar"] = range_diff
    return clusters_radar