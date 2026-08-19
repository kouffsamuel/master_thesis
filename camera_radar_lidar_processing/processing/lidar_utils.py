import argparse
import re
import sys

import numpy as np
import matplotlib

import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import cKDTree
# ==========================================
# CLUSTERING
# ==========================================

DBSCAN_EPS = 0.70          # m

DBSCAN_MIN_POINTS = 10

CLUSTER_COLORS = np.array([
    [0.00, 0.70, 0.00],   # Vert
    [0.00, 0.30, 1.00],   # Bleu
    [1.00, 0.00, 1.00],   # Magenta
    [0.00, 1.00, 1.00],   # Cyan
    [0.60, 0.00, 1.00],   # Violet
    [0.00, 0.00, 0.00],   # Noir
    [0.50, 0.25, 0.00],   # Marron
    [1.00, 0.40, 0.70],   # Rose
    [0.50, 0.50, 0.50],   # Gris
])


def filter_fov(pcd, hfov, vfov):

    pts = np.asarray(pcd.points)
    horizontal = np.degrees(np.arctan2(pts[:,1], pts[:,0]))

    vertical = np.degrees(
        np.arctan2(
            pts[:,2],
            np.sqrt(pts[:,0]**2 + pts[:,1]**2)
        )
    )

    mask = (
        (np.abs(horizontal) <= hfov / 2) &
        (np.abs(vertical) <= vfov / 2)
    )

    pcd.points = o3d.utility.Vector3dVector(pts[mask])

    return pcd

# ======================================
# OPEN PLY FILES
# ======================================
def load_lidar(path):
    """
        Load the lidar point cloud and apply a processing by Open3D

        STEP :
            1. Load the .ply
            2. Conversion mm -> m
            3. Crop the good area ( optional )
            4. Downsampling ( optional just to reduce the number of point if need ) =>  instead of this maybe clustering 
            5. Retour des coordonnées x, y, z
    """

    pcd = o3d.io.read_point_cloud(str(path))

    if pcd.is_empty():
        return None

    pts = np.asarray(pcd.points, dtype=np.float32)

    # mm -> m
    pts /= 1000.0

    pcd.points = o3d.utility.Vector3dVector(pts)

    # Optionnel
    # pcd = pcd.voxel_down_sample(voxel_size=0.05)

    if pcd.is_empty():
        return None

    return pcd


CMAP_HOT = LinearSegmentedColormap.from_list(
    "lidar_hot", ["#000000", "#3d0000", "#8b0000", "#e01b00", "#ff8c00", "#ffd24a"]
)


def colorize(pcd, keep, depth, mode="reflectivity", cmap=CMAP_HOT):
    xyz = np.asarray(pcd.points)
    has_colors = pcd.has_colors()

    if mode == "reflectivity" and has_colors:
        # On suppose que la réflectivité a été stockée dans le canal rouge des couleurs
        colors = np.asarray(pcd.colors)
        val = colors[keep, 0]
        lo, hi = np.percentile(val, [2, 98])
    elif mode == "height":
        val = xyz[keep, 2]
        lo, hi = np.percentile(val, [2, 98])
    else:  # depth (ou fallback si pas de réflectivité disponible)
        val = np.log10(depth)
        lo, hi = np.percentile(val, [2, 98])

    t = np.clip((val - lo) / max(hi - lo, 1e-9), 0, 1)
    rgba = cmap(t)

    # Atténuation atmosphérique : le lointain s'assombrit -> profondeur lisible.
    fade = np.clip(1.05 - 0.35 * (depth / np.percentile(depth, 97)), 0.35, 1.0)
    rgba[:, :3] *= fade[:, None]
    return rgba

#-------------------------------------
# BACKGROUND REMOVAL
#-------------------------------------
def voxelize_background(background, voxel_size=0.30):
    """
    Convertit un nuage de points en une représentation voxelisée.
    """

    # Indice du voxel auquel appartient chaque point
    voxel_idx = np.floor(background / voxel_size).astype(np.int32)

    # On ne garde qu'un voxel de chaque type
    voxel_idx = np.unique(voxel_idx, axis=0)

    # Centre de chaque voxel
    voxel_centers = (voxel_idx + 0.5) * voxel_size

    return voxel_centers


def build_background(lidar_files, n_background=10, voxel_size=0.30, alpha=0.3):
    voxel_occupancy = {}  
    voxel_max_occ = {}   

    for f in lidar_files[:n_background]:
        pcd = load_lidar(f)
        if pcd is None or pcd.is_empty():
            continue

        pts = np.asarray(pcd.points, dtype=np.float32)
        voxel_keys = set(map(tuple, np.floor(pts / voxel_size).astype(int)))

        for key in list(voxel_occupancy.keys()):
            voxel_occupancy[key] *= (1 - alpha)

        for key in voxel_keys:
            voxel_occupancy[key] = voxel_occupancy.get(key, 0.0) + alpha
            if voxel_occupancy[key] > voxel_max_occ.get(key, 0.0):
                voxel_max_occ[key] = voxel_occupancy[key]

    return voxel_max_occ 

def remove_background(pcd, voxel_occupancy, voxel_size=0.30, occupancy_threshold=0.6, check_neighbors=True):
    if pcd is None or pcd.is_empty():
        return o3d.geometry.PointCloud()

    pts = np.asarray(pcd.points, dtype=np.float32)
    voxel_keys = np.floor(pts / voxel_size).astype(int)

    if not check_neighbors:
        is_background = np.array([
            voxel_occupancy.get(tuple(k), 0.0) >= occupancy_threshold
            for k in voxel_keys
        ])
    else:
        # regarde aussi les voxels voisins directs (gère la dilution aux frontières)
        offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
        is_background = np.zeros(len(voxel_keys), dtype=bool)
        for i, k in enumerate(voxel_keys):
            for off in offsets:
                neighbor = (k[0]+off[0], k[1]+off[1], k[2]+off[2])
                if voxel_occupancy.get(neighbor, 0.0) >= occupancy_threshold:
                    is_background[i] = True
                    break

    filtered_pts = pts[~is_background]
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_pts)
    return filtered_pcd

#---------------------------------------------
# CLUSTERING
#---------------------------------------------

def clusterize(pcd):

    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return []

    labels = np.array(
        pcd.cluster_dbscan(
            eps=DBSCAN_EPS,
            min_points=DBSCAN_MIN_POINTS,
            print_progress=False
        )
    )

    clusters = []

    max_label = labels.max()

    if max_label < 0:
        return clusters

    for i in range(max_label + 1):

        idx = np.where(labels == i)[0]

        cluster = pcd.select_by_index(idx)

        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]

        cluster.paint_uniform_color(color)

        clusters.append({ "id": i, "pcd": cluster, "color": color, "center": cluster.get_center(), 
                         "aabb": cluster.get_axis_aligned_bounding_box(),"obb": cluster.get_oriented_bounding_box(),
                        "extent": cluster.get_oriented_bounding_box().extent,"n_points": len(cluster.points) })

    return clusters
