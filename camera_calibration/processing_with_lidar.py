import os
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2

# AJOUTE CES DEUX LIGNES :
import matplotlib

from mpl_toolkits.mplot3d import Axes3D   # (pas indispensable si on fait uniquement du BEV)
import open3d as o3d
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D  # à ajouter en haut du fichier

# matplotlib.use("Agg")
# # matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
# ==========================================
# PATHS
# ==========================================
data_path = "/DATA_MUSE"
folder_path = f"{data_path}/2026_08_07_10_05_00"
save_video = True

output_video = f"/home/skouff/video_lidar.mp4"
dir_raw = Path(f"{folder_path}/radar")
dir_camera = Path(f"{folder_path}/camera")
dir_lidar = Path(f"{folder_path}/lidar")
background = np.load("/home/skouff/background_db_with_plexi.npy")

# ==========================================
# LOAD FILES + TIMESTAMPS
# ==========================================
def load_files(folder, ext):
    files = sorted(folder.glob(f"*{ext}"))
    times = np.array([float(f.stem) for f in files])
    return files, times

raw_files, raw_times = load_files(dir_raw, ".raw")
raw_files = raw_files[1:]
cam_files, cam_times = load_files(dir_camera, ".jpeg")
lidar_files = sorted(dir_lidar.glob("*.ply"))
lidar_times = np.array([int(f.stem) / 1e9 for f in lidar_files])


# ==========================================
# UTILS
# ==========================================
def find_closest_index(times_array, target_time):
    return np.argmin(np.abs(times_array - target_time))


# ==========================================
# RADAR PARAMETERS AND PROCESSING
# ==========================================
c = 3e8             # Speed of light in m/s
fc = 24.125e9       # Frequency
lam = c / fc        # Wavelength in meters
BW = 554e6          # Bandwidth
N = 256             # Number of range bins
clk = 38461538      # Clock frequency
delay = 2214

delta_v = (lam * clk * 3.6) / (2 * N * (12 * (N + 4) + delay))
Vmax = delta_v * (N // 2 - 1)

range_bins = np.arange(N) * (c / (2 * BW))
velocity_bins = np.arange(N) * delta_v - Vmax

def get_complex_content(file):
    data = open(file, "rb").read()
    arr = np.frombuffer(data, dtype=np.uint16)

    content = np.empty((3, 256, 256), dtype="complex")
    size = 2 * 256 * 256
    for i in range(3):
        sub = arr[i*size:(i+1)*size]
        content[i] = (sub[0::2] + 1j * sub[1::2]).reshape((256, 256))
    return content

def FFT(RX):
    window = np.hamming(256)
    rx = RX * window[:, None]
    fft = np.fft.fft2(RX)
    rd = np.fft.fftshift(fft, axes=0)
    return np.abs(rd)


def compute_rd(file):
    content = get_complex_content(file)
    RX1 = FFT(content[0])
    RX2 = FFT(content[1])
    RX3 = FFT(content[2])

    RD_avg = (RX1 + RX2 + RX3) / 3
    magn = 20 * np.log10(RD_avg + 1e-6) - background
    rd_db = np.clip(magn, 0, None)
    return rd_db

# ==========================================
# LIDAR VIEW PARAMETERS
# ==========================================

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
YAW = 0
PITCH = 0
ROLL = 0
NEAR = 0
LIDAR_TO_CAM_TRANSLATION = np.array([0.0, 0.0, 0.20])

POINT_SIZE = 7.0
COLOR_MODE = "reflectivity"
BACKGROUND = "white"

CMAP_HOT = LinearSegmentedColormap.from_list(
    "lidar_hot", ["#000000", "#3d0000", "#8b0000", "#e01b00", "#ff8c00", "#ffd24a"]
)

# ==========================================
# LIDAR PROCESSING
# ==========================================
def filter_fov(pts, hfov, vfov):

    horizontal = np.degrees(np.arctan2(pts["y"], pts["x"]))

    vertical = np.degrees(
        np.arctan2(
            pts["z"],
            np.sqrt(pts["x"]**2 + pts["y"]**2)
        )
    )

    mask = (
        (np.abs(horizontal) <= hfov / 2) &
        (np.abs(vertical) <= vfov / 2)
    )

    return {k: v[mask] for k, v in pts.items()}

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
def rotation(yaw_deg=0, pitch_deg=0, roll_deg=0):
    ya, pi, ro = np.radians([yaw_deg, pitch_deg, roll_deg])
    Rz = np.array([[np.cos(ya), -np.sin(ya), 0], [np.sin(ya), np.cos(ya), 0], [0, 0, 1]])
    Ry = np.array([[np.cos(pi), 0, np.sin(pi)], [0, 1, 0], [-np.sin(pi), 0, np.cos(pi)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(ro), -np.sin(ro)], [0, np.sin(ro), np.cos(ro)]])
    return Rz @ Ry @ Rx

# Matrice de passage : (x=profondeur, y=latéral, z=vertical) -> convention OpenCV (X=droite, Y=bas, Z=profondeur)
AXES_LIDAR_TO_CV = np.array([
    [0, -1,  0],
    [0,  0, -1],
    [1,  0,  0],
], dtype=np.float64)

def project_real_camera(pts, camMatrix, distCoeff, width, height,
                         R=None, t=None, near=NEAR):

    if R is None:
        R = np.eye(3)
    if t is None:
        t = LIDAR_TO_CAM_TRANSLATION

    xyz = np.column_stack((pts["x"], pts["y"], pts["z"])).astype(np.float64)

    # Extrinsèque LiDAR -> caméra : rotation + translation
    P = (rotation() @ xyz.T).T + t.reshape(1, 3)

    forward = P[:, 0]
    front = forward > near
    P = P[front]
    forward = forward[front]

    if len(P) == 0:
        return (np.array([]), np.array([]), np.array([]), np.array([], dtype=int))

    # Passage vers la convention OpenCV pour projectPoints
    P_cv = (AXES_LIDAR_TO_CV @ P.T).T

    rvec = np.zeros((3, 1))
    tvec = np.zeros((3, 1))
    projected, _ = cv2.projectPoints(P_cv, rvec, tvec, camMatrix, distCoeff)
    u = projected[:, 0, 0]
    v = projected[:, 0, 1]

    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    keep = np.where(front)[0][inside]

    return u[inside], v[inside], forward[inside], keep


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

    for f in lidar_files[-n_background:]:

        pts = load_lidar(f)

        if pts is None:
            continue

        xyz = np.column_stack((
            pts["x"],
            pts["y"],
            pts["z"]
        ))

        background.append(xyz)

    background = np.vstack(background)

    background_voxels = voxelize_background(
        background,
        voxel_size=0.30
    )

    return background_voxels,background

def remove_background(points,
                      background_voxels,
                      voxel_size=0.30):
    """
    Remove background points based on voxel occupancy.
    """

    # Coordonnées des points
    xyz = np.column_stack((
        points["x"],
        points["y"],
        points["z"]
    ))

    # Voxel de chaque point de la frame
    point_voxels = np.floor(xyz / voxel_size).astype(np.int32)

    # Voxel du background
    background_idx = np.floor(
        background_voxels / voxel_size
    ).astype(np.int32)

    background_set = set(map(tuple, background_idx))

    # On conserve uniquement les points qui ne sont PAS dans un voxel du background
    mask = np.array(
        [tuple(v) not in background_set for v in point_voxels],
        dtype=bool
    )

    return {k: v[mask] for k, v in points.items()}


def xyz_to_points(xyz):

    return {
        "x": xyz[:,0],
        "y": xyz[:,1],
        "z": xyz[:,2],
        "reflectivity": np.ones(len(xyz))
    }


# ==========================================
# REAL-TIME VIEWER
# ==========================================
class RealTimeViewer:
    def __init__(self):
        self.paused = False
        self.background_voxels,self.background = build_background(lidar_files, n_background=50)
        self.calibration = np.load("/home/skouff/master_thesis/camera_calibration/calibration_with_square_size.npz")
        self.camMatrix = self.calibration["camMatrix"]       
        self.distCoeff = self.calibration["distCoeff"]        

        # CREATE THE PLOT DIVISION AND AXES#
        self.fig, (self.ax_rd, self.ax_cam) = plt.subplots(1, 2, figsize=(15, 6))

        self.im_rd = None
        self.cbar = None
        self.im_cam = None
        self.scatter_lidar = None

        self.title_rd = self.ax_rd.set_title("")

        self.ax_rd.set_xlabel("Velocity (km/h)")
        self.ax_rd.set_ylabel("Range (m)")

        self.ax_cam.axis("off")

        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        plt.ion()
        plt.show()

    def on_key(self, event):
        if event.key == 'p':
            self.paused = not self.paused
            print("Paused" if self.paused else "Resuming")
        elif event.key == 'q':
            print("Quitting...")
            plt.close('all')
            os._exit(0)

    def update(self, i):

        "update the window to read through all the dataset"

        raw_file = raw_files[i]
        t = raw_times[i]

        # ================= RADAR =================
        rd = compute_rd(raw_file)

        if self.im_rd is None:
            self.im_rd = self.ax_rd.imshow(
                rd.T,
                extent=[velocity_bins[0], velocity_bins[-1], range_bins[0], range_bins[-1]],
                origin='lower',
                cmap='gray_r',
                vmin=0,
                aspect='auto'
            )
            self.cbar = self.fig.colorbar(self.im_rd, ax=self.ax_rd)
            self.cbar.set_label("dB")
        else:
            self.im_rd.set_data(rd.T)

        self.title_rd.set_text(f"Radar t = {t:.6f}")

        # ================= CAMERA =================
        idx = find_closest_index(cam_times, t)
        img = np.array(Image.open(cam_files[idx]))

        if self.im_cam is None:
            self.im_cam = self.ax_cam.imshow(img)
        else:
            self.im_cam.set_data(img)

        self.ax_cam.set_title(f"Camera t = {cam_times[idx]:.6f}")

        # ================= LIDAR (superposé sur la caméra) =================
        idx_lidar = find_closest_index(lidar_times, t)
        pts = load_lidar(lidar_files[idx_lidar])

        if pts is not None:
            pts = remove_background(pts, self.background_voxels)
            u, v, forward, keep = project_real_camera(pts, self.camMatrix, self.distCoeff, IMAGE_WIDTH, IMAGE_HEIGHT)


            if len(u) > 0:
                rgba = colorize(pts, keep, forward, COLOR_MODE)
                size = np.clip(POINT_SIZE * 26.0 / forward, 0.7, 60.0)

                if self.scatter_lidar is None:
                    self.scatter_lidar, = self.ax_cam.plot(u, v, 'r.' , markersize=2)
                else:
                    self.scatter_lidar.set_data(u,v)
        # refresh rapide
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        # pause if asked
        while self.paused:
            plt.pause(0.05)    

# ==========================================
# MAIN
# ==========================================
def main():
    viewer = RealTimeViewer()
    writer = FFMpegWriter(fps=15)
    print(folder_path)
    print(os.path.isdir(folder_path))

    with writer.saving(viewer.fig, output_video, dpi=200):
        for i in range(len(raw_files)):
            viewer.update(i)
            if save_video:
                writer.grab_frame()

            plt.pause(0.001)

    plt.ioff()
    plt.show()
    print("Video saved:", output_video)

if __name__ == "__main__":
    main()

