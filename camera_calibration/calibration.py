import cv2
import numpy as np
from pathlib import Path
import glob
import os


def calibrate():
    cam_files = glob.glob(os.path.join("/DATA_MUSE/camera_calibration_files/camera/", "*.jpeg"))

    # Init checkerboard
    nRows = 7
    nCols = 10
    termCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)

    # Initialize world point coordinates (x,y,z=0)
    worldPtsCur = np.zeros((nRows*nCols, 3), np.float32)

    # Create 2D grid and flatten to get a list of peers 
    squareSize = 0.02
    worldPtsCur[:, :2] = np.mgrid[0:nCols, 0:nRows].T.reshape(-1, 2) * squareSize
    worldPtsList = []
    imgPtsList = []

    cv2.namedWindow('ChessBoard', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('ChessBoard', 960, 540)

    for img in cam_files:
        imgBGR = cv2.imread(img)
        imgGray = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2GRAY)
        cornersFound, cornersOrg = cv2.findChessboardCorners(imgGray, (nCols, nRows), None)

        if cornersFound == True:
            worldPtsList.append(worldPtsCur)
            cornersRefined = cv2.cornerSubPix(imgGray, cornersOrg, (11,11), (-1,-1), termCriteria)
            imgPtsList.append(cornersRefined)

            cv2.drawChessboardCorners(imgBGR, (nCols, nRows), cornersRefined, cornersFound)
            cv2.imshow('ChessBoard', imgBGR)
            cv2.waitKey(500)
    cv2.destroyAllWindows()

    # Calibrate
    repError, camMatrix, distCoeff, rvecs, tvecs = cv2.calibrateCamera(worldPtsList, imgPtsList, imgGray.shape[::-1], None, None)
    print("Camera Matrix:\n", camMatrix)
    print("Reproj Error (pixels): {:.4f}".format(repError))

    # Save Calibration
    param_path = os.path.join("/home/skouff/master_thesis/camera_calibration", "calibration_with_square_size.npz")
    np.savez(param_path, repError=repError, camMatrix=camMatrix, distCoeff=distCoeff, rvecs=rvecs, tvecs=tvecs)

if __name__ == "__main__":
    calibrate()