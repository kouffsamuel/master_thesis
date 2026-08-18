#!/usr/bin/env python3

import argparse
import re
import sys

import numpy as np
import matplotlib

import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import cKDTree

from master_thesis.camera_radar_lidar_processing.processing.lidar_utils import build_background, clusterize, colorize, filter_fov, load_lidar, remove_background



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
    pcd = load_lidar(lidar_files[FRAME_INDEX])
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





