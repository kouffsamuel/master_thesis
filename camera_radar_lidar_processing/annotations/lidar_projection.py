import numpy as np
import cv2

NEAR = 0
LIDAR_TO_CAM_TRANSLATION = np.array([0.0, -0.02, 0.18])
LIDAR_TO_RADAR_TRANSLATION = np.array([0.0, -0.01, 0.10])


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

def project_real_camera(pcd, camMatrix, distCoeff, width, height,
                         R=None, t=None, near=NEAR):

    if R is None:
        R = rotation(-1.5,0,0)
    if t is None:
        t = LIDAR_TO_CAM_TRANSLATION

    xyz = np.asarray(pcd.points, dtype=np.float64)

    if xyz.shape[0] == 0:
        return (np.array([]), np.array([]), np.array([]), np.array([], dtype=int))

    # Extrinsèque LiDAR -> caméra : rotation + translation
    forward = (R @ xyz.T).T[:, 0]
    front = forward > near
    xyz = xyz[front]
    forward = forward[front]

    if len(xyz) == 0:
        return (np.array([]), np.array([]), np.array([]), np.array([], dtype=int))

    # Passage vers la convention OpenCV pour projectPoints
    R_total = AXES_LIDAR_TO_CV @ R
    t_total = AXES_LIDAR_TO_CV @ t
    rvec, _ = cv2.Rodrigues(R_total)

    projected, _ = cv2.projectPoints(xyz, rvec, t_total, camMatrix, distCoeff)
    u = projected[:, 0, 0]
    v = projected[:, 0, 1]

    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    keep = np.where(front)[0][inside]

    return u[inside], v[inside], forward[inside], keep


def project_point_to_camera(center_xyz, camMatrix, distCoeff, R=None, t=None):
    """
    Projette un point 3D (centre de cluster) dans l'image caméra.
    Retourne (u, v) ou None si le point est derrière la caméra.
    """
    if R is None:
        R = rotation(-1.5,0,0)
    if t is None:
        t = LIDAR_TO_CAM_TRANSLATION

    xyz = np.asarray(center_xyz, dtype=np.float64).reshape(1, 3)

    forward = (R @ xyz.T).T[:, 0]
    if forward[0] <= NEAR:
        return None

    R_total = AXES_LIDAR_TO_CV @ R
    t_total = AXES_LIDAR_TO_CV @ t
    rvec, _ = cv2.Rodrigues(R_total)

    projected, _ = cv2.projectPoints(xyz, rvec, t_total, camMatrix, distCoeff)
    u, v = projected[0, 0]
    return (float(u), float(v))

def lidar_to_radar_range(centroid, t=None):
    if t is None:
        t = LIDAR_TO_RADAR_TRANSLATION

    xyz = np.asarray(centroid, dtype=np.float64)
    xyz_radar = xyz + t
    range_meters = np.sqrt(np.sum(xyz_radar **2))
    return range_meters