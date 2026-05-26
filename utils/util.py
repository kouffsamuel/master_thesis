import torch
import numpy as np
import cv2
from shapely.geometry import Polygon
from shapely.ops import unary_union
import polarTransform
from utils.metrics import process_predictions_FFT

# Camera parameters
camera_matrix = np.array([[1.84541929e+03, 0.00000000e+00, 8.55802458e+02],
                 [0.00000000e+00 , 1.78869210e+03 , 6.07342667e+02],[0.,0.,1]])
dist_coeffs = np.array([2.51771602e-01,-1.32561698e+01,4.33607564e-03,-6.94637533e-03,5.95513933e+01])
# Represents the camera's orientation in 3D space (Rodrigues format)
rvecs = np.array([1.61803058, 0.03365624,-0.04003127])
# The position (in metres) of the camera in the world coordinate system 
tvecs = np.array([0.09138029,1.38369885,1.43674736])
ImageWidth = 1920
ImageHeight = 1080

CLASS_TO_ID = {0: 'car', 1: 'truck', 2: 'bicycle', 3: 'bus', 4: 'person'}

def worldToImage(x,y,z):
    world_points = np.array([[x,y,z]],dtype = 'float32')
    rotation_matrix = cv2.Rodrigues(rvecs)[0]

    imgpts, _ = cv2.projectPoints(world_points, rotation_matrix, tvecs, camera_matrix, dist_coeffs)

    u = min(max(0,imgpts[0][0][0]),ImageWidth-1)
    v = min(max(0,imgpts[0][0][1]),ImageHeight-1)

    return u,v

def imageToWorld(u, v, z_world=0.0):
    """
    Convert image pixel (u, v) into world coordinates (X, Y, Z)
    assuming the point lies on a plane Z = z_world (default = ground plane).
    """

    # Step 1: Undistort the pixel
    pts = np.array([[[u, v]]], dtype=np.float32)
    undistorted = cv2.undistortPoints(pts, camera_matrix, dist_coeffs)
    x_norm, y_norm = undistorted[0][0]

    # Step 2: Ray in camera coordinates
    ray_cam = np.array([x_norm, y_norm, 1.0])

    # Step 3: Rotation matrix
    R, _ = cv2.Rodrigues(rvecs)

    # Step 4: Camera center in world coordinates
    C = -R.T @ tvecs

    # Step 5: Ray direction in world coordinates
    ray_world = R.T @ ray_cam

    # Step 6: Intersect ray with plane Z = z_world
    if abs(ray_world[2]) < 1e-6:
        raise ValueError("Ray is parallel to the plane")

    t = (z_world - C[2]) / ray_world[2]

    world_point = C + t * ray_world

    return world_point[0], world_point[1], world_point[2]

def DisplayHMI(image, input, box_labels, model_outputs,encoder,config,intermediate=None):
    image_copy = image.copy()
    # Model outputs
    pred_obj = model_outputs['Detection'].detach().cpu().numpy().copy()[0]

    # Decode the output detection map
    pred_obj = encoder.decode(pred_obj,0.05)
    pred_obj = np.asarray(pred_obj)

    # process prediction: polar to cartesian, NMS...
    if(len(pred_obj)>0):
        pred_obj = process_predictions_FFT(pred_obj,confidence_threshold=0.5)

    ## FFT
    if config['data_mode'] != 'ADC':
        FFT = np.abs(input[...,:16]+input[...,16:]*1j).mean(axis=2)
    else:
        FFT = np.abs(intermediate[:16,:,:]+intermediate[16:,:,:]*1j).mean(axis=0)

    PowerSpectrum = np.log10(FFT)
    # rescale
    PowerSpectrum = (PowerSpectrum -PowerSpectrum.min())/(PowerSpectrum.max()-PowerSpectrum.min())*255
    PowerSpectrum = cv2.cvtColor(PowerSpectrum.astype('uint8'),cv2.COLOR_GRAY2BGR)

    ## Image
    for box in pred_obj:
        box = box[1:] # Keep 8 coordinates (+ range and angle), remove confidence score
        # box = [x1, y1, x2, y2, x3, y3, x4, y4, angle, range]
        u1,v1 = worldToImage(-box[2],box[1],0)
        u2,v2 = worldToImage(-box[0],box[1],1.6)

        u1 = int(u1/2)
        v1 = int(v1/2)
        u2 = int(u2/2)
        v2 = int(v2/2)

        image_copy = cv2.rectangle(image_copy, (u1,v1), (u2,v2), (0, 0, 255), 1)
        
    ## Plotting GT on image
    for box in box_labels:
        box_coord = box[6:10] # Keep coordinates
        class_id = int(box[10])
        class_name = CLASS_TO_ID.get(class_id, str(class_id))

        u1 = int(box_coord[0]/2)
        v1 = int(box_coord[1]/2)
        u2 = int(box_coord[2]/2)
        v2 = int(box_coord[3]/2)

        image_copy = cv2.rectangle(image_copy, (u1,v1), (u2,v2), (0, 255, 0), 1)

        # Label au-dessus de la bounding box
        label_pos = (u1, max(v1 - 5, 10))  # évite de sortir du bord haut
        cv2.putText(
            image_copy,
            class_name,
            label_pos,
            cv2.DEJA,
            fontScale=0.5,
            color=(0, 0, 255),
            thickness=1,
            lineType=cv2.LINE_AA
        )


    return np.hstack((PowerSpectrum,image_copy[:512]))