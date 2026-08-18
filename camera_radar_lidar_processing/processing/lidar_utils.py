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
# DISPLAY FIELD OF VIEW
# ==========================================
DISPLAY_HFOV = 70.0      # <= 120°
DISPLAY_VFOV = 20.0      # <= 25°


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


def build_background(lidar_files, n_background=10):

    background = []

    #for f in lidar_files[-n_background:]:
    for f in lidar_files[:n_background]:    

        pcd = load_lidar(f)
        if pcd is None:
            continue

        # pcd = filter_fov( pcd , DISPLAY_HFOV,DISPLAY_VFOV)
        if pcd.is_empty():
            continue
    
        pts = np.asarray(pcd.points, dtype=np.float32)

        background.append(pts)

    background = np.vstack(background)

    background_voxels = voxelize_background(
        background,
        voxel_size=0.30
    )
    background_tree = cKDTree(background_voxels)

    return background_voxels, background_tree,background


def remove_background(pcd, background_voxels, voxel_size=0.30):
    """
    Remove background points based on voxel occupancy.
    """

    pts = np.asarray(pcd.points)

    if len(pts) == 0:
        return pcd

    # Voxel de chaque point de la frame
    point_voxels = np.floor(pts / voxel_size).astype(np.int32)

    # Voxel du background
    background_idx = np.floor(background_voxels / voxel_size).astype(np.int32)

    background_set = set(map(tuple, background_idx))

    # Conserver uniquement les points hors du background
    keep = [
        i for i, voxel in enumerate(point_voxels)
        if tuple(voxel) not in background_set ]

    return pcd.select_by_index(keep)

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
