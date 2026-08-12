    #!/usr/bin/env python3

import argparse
import re
import sys

import numpy as np
import matplotlib

import open3d as o3d
matplotlib.use("Agg")
matplotlib.use("TkAgg")
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

# ==========================================
# DATA
# ==========================================

from pathlib import Path

LIDAR_FOLDER = Path("/DATA_MUSE/2026_08_07_10_05_00/lidar")

lidar_files = sorted(LIDAR_FOLDER.glob("*.ply"))
lidar_times = np.array([int(f.stem) / 1e9 for f in lidar_files])

# ==========================================
# VIRTUAL CAMERA
# ==========================================

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
YAW = 0
PITCH = 0
ROLL = 0
NEAR = 0.5

LIDAR_TO_CAM_TRANSLATION = np.array([0.0, 0.0, 0.20])

# ==========================================
# DISPLAY
# ==========================================
POINT_SIZE = 2.0
COLOR_MODE = "reflectivity"
BACKGROUND = "white"

# ==========================================
# FILTERS
# ==========================================
MAX_RANGE = 80.0
DROP_NOISY = False

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
def read_ply(path):

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

# --------------------------------------------------------------------------
# PROJECTION DANS CAMERA VIRTUELLE
# --------------------------------------------------------------------------
def rotation(yaw_deg, pitch_deg, roll_deg):
    ya, pi, ro = np.radians([yaw_deg, pitch_deg, roll_deg])
    Rz = np.array([[np.cos(ya), -np.sin(ya), 0], [np.sin(ya), np.cos(ya), 0], [0, 0, 1]])
    Ry = np.array([[np.cos(pi), 0, np.sin(pi)], [0, 1, 0], [-np.sin(pi), 0, np.cos(pi)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(ro), -np.sin(ro)], [0, np.sin(ro), np.cos(ro)]])
    return Rz @ Ry @ Rx

def project_real_camera(pts, camMatrix, width, height, eye=(0, 0, 0),
            yaw=YAW, pitch=PITCH, roll=ROLL, near=NEAR, lidar_to_cam_t=LIDAR_TO_CAM_TRANSLATION):

    # Translation
    P = pts + lidar_to_cam_t

    # Orientation
    P = P - np.asarray(eye, float)
    P = P @ rotation(yaw, pitch, roll)

    forward = P[:, 0]
    front = forward > near
    P = P[front]
    forward = forward[front]

    fx = camMatrix[0, 0]
    fy = camMatrix[1, 1]
    cx = camMatrix[0, 2]
    cy = camMatrix[1, 2]

    u = cx - fx * P[:, 1] / forward
    v = cy - fy * P[:, 2] / forward

    inside = ((u >= 0) & (u < width) & (v >= 0) & (v < height))
    keep = np.where(front)[0][inside]
    return (u[inside], v[inside], forward[inside], keep)

# --------------------------------------------------------------------------
# 4. Colorisation
# --------------------------------------------------------------------------

CMAP_HOT = LinearSegmentedColormap.from_list(
    "lidar_hot", ["#000000", "#3d0000", "#8b0000", "#e01b00", "#ff8c00", "#ffd24a"]
)


def colorize(pts, keep, depth, mode="reflectivity", cmap=CMAP_HOT):
    if mode == "reflectivity" and "reflectivity" in pts:
        val = pts["reflectivity"][keep]
        lo, hi = np.percentile(val, [2, 98])
    elif mode == "height":
        val = pts[:,2][keep]
        lo, hi = np.percentile(val, [2, 98])
    else:  # depth
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

        pcd = read_ply(f)
        if pcd is None:
            continue

        pcd = filter_fov( pcd , DISPLAY_HFOV,DISPLAY_VFOV)
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


def render_clusters(ax, foreground, camMatrix, clusters):

    # --------------------------------------------------
    # Projection du foreground
    # --------------------------------------------------

    u, v, forward, keep = project_real_camera(foreground, camMatrix, IMAGE_WIDTH,IMAGE_HEIGHT)

    if len(u) == 0:
        ax.clear()
        ax.set_title("Clusters")
        ax.set_facecolor(BACKGROUND)
        ax.axis("off")
        return

     # --------------------------------------------------
    # Aucun cluster détecté
    # --------------------------------------------------

    if len(clusters) == 0:

        print("no clusteeeeers")

        margin = 20

        ax.set_xlim(u.min() - margin, u.max() + margin)
        ax.set_ylim(v.max() + margin, v.min() - margin)

        ax.set_aspect("equal")
        ax.set_facecolor(BACKGROUND)
        ax.axis("off")
        ax.set_title("Clusters (0 detected)", color="orange")

        return

    rgba = colorize( foreground,keep,forward,COLOR_MODE)

    size = np.clip(POINT_SIZE * 26.0 / forward,0.7,60.0)

    order = np.argsort(-forward)

    ax.clear()

    ax.scatter(
        u[order],
        v[order],
        s=size[order],
        c=rgba[order],
        marker=".",
        linewidths=0
    )

    # Limites de l'affichage
    u_min = u.min()
    u_max = u.max()

    v_min = v.min()
    v_max = v.max()


    for cluster in clusters:

        pts = np.asarray(cluster["pcd"].points,dtype=np.float32)
        u, v, forward, keep = project_real_camera(pts, camMatrix, IMAGE_WIDTH,IMAGE_HEIGHT)

        if len(u) == 0:
            continue

        rgba = np.tile(cluster["color"],(len(u), 1))

        size = np.clip(POINT_SIZE * 26.0 / forward,0.7,60.0)

        order = np.argsort(-forward)

        ax.scatter(
            u[order],
            v[order],
            s=size[order],
            c=rgba[order],
            marker=".",
            linewidths=0
        )

        u_min = min(u_min, u.min())
        u_max = max(u_max, u.max())

        v_min = min(v_min, v.min())
        v_max = max(v_max, v.max())

    # --------------------------------------------------
    # Configuration de l'affichage
    # --------------------------------------------------

    margin = 20

    ax.set_xlim(u_min - margin, u_max + margin)
    ax.set_ylim(v_max + margin, v_min - margin)

    ax.set_aspect("equal")

    ax.set_facecolor(BACKGROUND)

    ax.axis("off")

    ax.set_title("Clusters", color="orange")

# --------------------------------------------------------------------------
# 5. Render
# --------------------------------------------------------------------------


def render(ax, u, v, forward, rgba, width, height,
           point_size=1.0, background="white", label=None):

    # Taille des points (perspective)
    size = np.clip(point_size * 26.0 / forward, 0.7, 60.0)

    # Les points lointains sont dessinés en premier
    order = np.argsort(-forward)

    ax.clear()

    ax.scatter(
        u[order],
        v[order],
        s=size[order],
        c=rgba[order],
        marker=".",
        linewidths=0,
        rasterized=True
    )

    #ax.set_xlim(0, width)
    #ax.set_ylim(height, 0)
    if len(u) == 0:
        ax.clear()
        ax.set_title(label)
        ax.axis("off")
        return

    margin = 20

    ax.set_xlim(u.min() - margin, u.max() + margin)
    ax.set_ylim(v.max() + margin, v.min() - margin)

    ax.set_aspect("equal")
    ax.set_facecolor(background)
    ax.axis("off")

    if label is not None:
        ax.set_title(label, color="orange")

    return len(u)

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main():

    FRAME_INDEX = 105
    pcd = read_ply(lidar_files[FRAME_INDEX])
    pcd = filter_fov( pcd, DISPLAY_HFOV,DISPLAY_VFOV )
    pts = np.asarray(pcd.points, dtype=np.float32)
    print(pts.shape)

    # -------------------------------
    # Construction du background
    # -------------------------------

    background_voxels,background_tree,background = build_background(lidar_files, n_background=50)

    foreground = remove_background(pcd, background_voxels)
    clusters = clusterize(foreground)

    print("\n========== Clustering ==========")
    print(f"Number of clusters : {len(clusters)}")

    if len(clusters) == 0:
        print("No cluster detected.")
    else:
        for cluster in clusters:

            center = cluster["center"]

            print(
                f"Cluster {cluster['id']:2d} | "
                f"Center = ({center[0]:6.2f}, {center[1]:6.2f}, {center[2]:6.2f}) m | "
                f"Points = {cluster['n_points']}"
            )

    print("================================\n")

    foreground = np.asarray(foreground.points, dtype=np.float32)

    # Conversion des voxels pour affichage
    datasets = [pts, background, foreground, clusters]
    titles = ["Original frame","Background","Background removed","Clusters"]

    # -------------------------------
    # Figure
    # -------------------------------

    fig, axs = plt.subplots(
        2,
        2,
        figsize=(14,8)
    )

    fig.canvas.manager.set_window_title("Processing lidar data")

    data = np.load("calibration.npz")
    camMatrix = data["camMatrix"]

    for i, ax in enumerate(axs.ravel()):

        cloud = datasets[i]
        title = titles[i]

        if i == 0:
            # Original frame
            #print("je suis ici")
            u, v, forward, keep = project_real_camera(cloud, camMatrix, IMAGE_WIDTH, IMAGE_HEIGHT)
            rgba = colorize(cloud,keep,forward,COLOR_MODE)
            render(ax,u,v,forward,rgba,IMAGE_WIDTH,IMAGE_HEIGHT,POINT_SIZE,BACKGROUND,title)

        elif i == 1:
            # Background
            #print("je suis ici")
            u, v, forward, keep = project_real_camera(cloud, camMatrix, IMAGE_WIDTH, IMAGE_HEIGHT)
            rgba = colorize(cloud,keep,forward,COLOR_MODE)
            render(ax,u,v,forward,rgba,IMAGE_WIDTH,IMAGE_HEIGHT,POINT_SIZE,BACKGROUND,title)        

        elif i == 2:
            # Foreground
            #print("je suis ici")
            u, v, forward, keep = project_real_camera(cloud, camMatrix, IMAGE_WIDTH, IMAGE_HEIGHT)
            rgba = colorize(cloud,keep,forward,COLOR_MODE)
            render(ax,u,v,forward,rgba,IMAGE_WIDTH,IMAGE_HEIGHT,POINT_SIZE,BACKGROUND,title)        

        elif i == 3:
            #clusters
            print("je suis ici")
            print(len(clusters))
            render_clusters(ax,foreground, camMatrix, clusters)


    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()





