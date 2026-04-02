import torch
import numpy as np
import cv2
from shapely.geometry import Polygon
from shapely.ops import unary_union
import polarTransform

# Camera parameters
camera_matrix = np.array([[1.84541929e+03, 0.00000000e+00, 8.55802458e+02],
                 [0.00000000e+00 , 1.78869210e+03 , 6.07342667e+02],[0.,0.,1]])
dist_coeffs = np.array([2.51771602e-01,-1.32561698e+01,4.33607564e-03,-6.94637533e-03,5.95513933e+01])
rvecs = np.array([1.61803058, 0.03365624,-0.04003127])
tvecs = np.array([0.09138029,1.38369885,1.43674736])
ImageWidth = 1920
ImageHeight = 1080

def worldToImage(x,y,z):
    world_points = np.array([[x,y,z]],dtype = 'float32')
    rotation_matrix = cv2.Rodrigues(rvecs)[0]

    imgpts, _ = cv2.projectPoints(world_points, rotation_matrix, tvecs, camera_matrix, dist_coeffs)

    u = int(min(max(0,imgpts[0][0][0]),ImageWidth-1))
    v = int(min(max(0,imgpts[0][0][1]),ImageHeight-1))

    return u,v

def RA_to_cartesian_box(data):
    L = 4
    W = 1.8

    boxes = []
    for i in range(len(data)):

        x = np.sin(np.radians(data[i][1])) * data[i][0]
        y = np.cos(np.radians(data[i][1])) * data[i][0]

        boxes.append([x - W/2,y,x + W/2,y, x + W/2,y+L,x - W/2,y+L,data[i][0],data[i][1]])

    return boxes

def perform_nms(valid_class_predictions, valid_box_predictions, nms_threshold):
    # Remove redudant detections. Strongly overlapping boxes lower-confidence boxes are supressed
    # while isolated or weakly overlapping boxes remain. The same mask is applied to both box
    # and score arrays so theu stay aligned. 

    # sort the detections such that the entry with the maximum confidence score is at the top
    sorted_indices = np.argsort(valid_class_predictions)[::-1]
    sorted_box_predictions = valid_box_predictions[sorted_indices]
    sorted_class_predictions = valid_class_predictions[sorted_indices]

    for i in range(sorted_box_predictions.shape[0]):
        # get the IOUs of all boxes with the currently most certain bounding box
        try:
            ious = np.zeros((sorted_box_predictions.shape[0]))
            ious[i + 1:] = bbox_iou(sorted_box_predictions[i, :8], sorted_box_predictions[i + 1:, :8])
        except ValueError:
            break
        except IndexError:
            break

        # eliminate all detections which have IoU > threshold
        overlap_mask = np.where(ious < nms_threshold, True, False)
        sorted_box_predictions = sorted_box_predictions[overlap_mask]
        sorted_class_predictions = sorted_class_predictions[overlap_mask]

    return sorted_class_predictions, sorted_box_predictions

def bbox_iou(box1, boxes):

    # currently inspected box
    box1 = box1.reshape((4,2))
    rect_1 = Polygon([(box1[0, 0], box1[0, 1]), (box1[1, 0], box1[1, 1]), (box1[2, 0], box1[2, 1]),
                      (box1[3, 0], box1[3, 1])])
    area_1 = rect_1.area

    # IoU of box1 with each of the boxes in "boxes"
    ious = np.zeros(boxes.shape[0])
    for box_id in range(boxes.shape[0]):
        box2 = boxes[box_id]
        box2 = box2.reshape((4,2))
        rect_2 = Polygon([(box2[0, 0], box2[0, 1]), (box2[1, 0], box2[1, 1]), (box2[2, 0], box2[2, 1]),
                          (box2[3, 0], box2[3, 1])])
        area_2 = rect_2.area

        # get intersection of both bounding boxes
        inter_area = rect_1.intersection(rect_2).area

        # compute IoU of the two bounding boxes
        iou = inter_area / (area_1 + area_2 - inter_area)

        ious[box_id] = iou

    return ious


def process_predictions_FFT(batch_predictions, confidence_threshold=0.1, nms_threshold=0.05):

    # process targets and perform NMS for each prediction in batch
    final_batch_predictions = None  # store final bounding box predictions

    point_cloud_reg_predictions = RA_to_cartesian_box(batch_predictions)
    point_cloud_reg_predictions = np.asarray(point_cloud_reg_predictions)
    point_cloud_class_predictions = batch_predictions[:,-1]

    # get valid detections
    validity_mask = np.where(point_cloud_class_predictions > confidence_threshold, True, False)
    valid_box_predictions = point_cloud_reg_predictions[validity_mask]
    valid_class_predictions = point_cloud_class_predictions[validity_mask]

    # perform Non-Maximum Suppression
    final_class_predictions, final_box_predictions = perform_nms(valid_class_predictions, valid_box_predictions,nms_threshold)

    # concatenate point_cloud_id, confidence score and bounding box prediction | shape: [N_FINAL, 1+1+8]
    final_point_cloud_predictions = np.hstack((final_class_predictions[:, np.newaxis],
                                               final_box_predictions))

    return final_point_cloud_predictions

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
        # box = [x1, y1, x2, y2, x3, y3, x4, y4, range, angle]
        u1,v1 = worldToImage(-box[2],box[1],0)
        u2,v2 = worldToImage(-box[0],box[1],1.6)

        u1 = int(u1/2)
        v1 = int(v1/2)
        u2 = int(u2/2)
        v2 = int(v2/2)

        image_copy = cv2.rectangle(image_copy, (u1,v1), (u2,v2), (0, 0, 255), 2)
        
    ## Plotting GT on image
    for box in box_labels:
        box = box[-4:] # Keep coordinates

        u1 = int(box[0]/2)
        v1 = int(box[1]/2)
        u2 = int(box[2]/2)
        v2 = int(box[3]/2)

        image_copy = cv2.rectangle(image_copy, (u1,v1), (u2,v2), (0, 255, 0), 2)


    return np.hstack((PowerSpectrum,image_copy[:512]))