import numpy as np
import open3d as o3d
from matplotlib.colors import LinearSegmentedColormap

#==========================================
# LIDAR VIEW PARAMETERS
# ==========================================

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
YAW = 0
PITCH = 0
ROLL = 0
NEAR = 0.5

POINT_SIZE = 7.0
COLOR_MODE = "reflectivity"
BACKGROUND = "white"

CMAP_HOT = LinearSegmentedColormap.from_list(
    "lidar_hot", ["#000000", "#3d0000", "#8b0000", "#e01b00", "#ff8c00", "#ffd24a"]
)

DISPLAY_HFOV = 64     # <= 120°
DISPLAY_VFOV = 38      # <= 25°

# ==========================================
# LIDAR PROCESSING
# ==========================================

def load_lidar(file):
    """
    Load the lidar point cloud and apply a processing by Open3D

    STEP :
        1. Load the .ply
        2. Conversion mm -> m
        3. Crop the good area ( optional )
        4. Downsampling ( optional just to reduce the number of point if need ) =>  instead of this maybe clustering 
        5. Retour des coordonnées x, y, z
    """

    # Files reading
    pcd = o3d.io.read_point_cloud(str(file))

    # Conversion to  NumPy
    pts = np.asarray(pcd.points, dtype=np.float32)

    if pts.shape[0] == 0:
        return None

    # Conversion mm -> m
    pts /= 1000.0
    pcd.points = o3d.utility.Vector3dVector(pts)

    # Point number reduction
    #pcd = pcd.voxel_down_sample(voxel_size=0.05)

    pts = np.asarray(pcd.points)

    if pts.shape[0] == 0:
        return None, None, None

    return  { "x": pts[:,0], "y": pts[:,1], "z": pts[:,2] }


# ==========================================
# LIDAR PROJECTION
# ==========================================
def rotation(yaw_deg, pitch_deg, roll_deg):
    ya, pi, ro = np.radians([yaw_deg, pitch_deg, roll_deg])
    Rz = np.array([[np.cos(ya), -np.sin(ya), 0], [np.sin(ya), np.cos(ya), 0], [0, 0, 1]])
    Ry = np.array([[np.cos(pi), 0, np.sin(pi)], [0, 1, 0], [-np.sin(pi), 0, np.cos(pi)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(ro), -np.sin(ro)], [0, np.sin(ro), np.cos(ro)]])
    return Rz @ Ry @ Rx

def project(pts,width,height,hfov_deg=120.0,vfov_deg=25.0,eye=(0, 0, 0),
            yaw=0.0,pitch=0.0,roll=0.0,near=0.5,):

    P = np.stack(
        [pts["x"], pts["y"], pts["z"]],
        axis=1
    ) - np.asarray(eye, float)

    P = P @ rotation(yaw, pitch, roll)


    forward = P[:, 0]
    front = forward > near
    P = P[front]
    forward = forward[front]

    fx = (width / 2.0) / np.tan(np.deg2rad(hfov_deg / 2.0))
    fy = (height / 2.0) / np.tan(np.deg2rad(vfov_deg / 2.0))

    u = width / 2.0 - fx * P[:, 1] / forward
    v = height / 2.0 - fy * P[:, 2] / forward

    inside = ( (u >= 0) & (u < width) & (v >= 0) & (v < height))
    keep = np.where(front)[0][inside]
    return ( u[inside],  v[inside],  forward[inside],  keep)


def colorize(pts, keep, depth, mode="reflectivity", cmap=CMAP_HOT):
    if mode == "reflectivity" and "reflectivity" in pts:
        val = pts["reflectivity"][keep]
        lo, hi = np.percentile(val, [2, 98])
    elif mode == "height":
        val = pts["z"][keep]
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

