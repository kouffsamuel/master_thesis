import cv2
import numpy as np
from pathlib import Path
import glob
import os


def calibrate():
    cam_files = glob.glob(os.path.join("/DATA_MUSE/calibration_files_plexi/", "*.jpg"))

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
    # width, height = 1920, 1080
    # hfov_guess_deg = 62.2
    # vfov_guess_deg = 48.8

    # fx_guess = width  / (2 * np.tan(np.radians(hfov_guess_deg) / 2))
    # fy_guess = height / (2 * np.tan(np.radians(vfov_guess_deg) / 2))

    # cam_matrix_guess = np.array([
    #     [fx_guess, 0,        width / 2],
    #     [0,        fy_guess, height / 2],
    #     [0,        0,        1]
    # ], dtype=np.float64)
    # dist_coeffs_guess = np.zeros(5, dtype=np.float64)  # ou tes coeffs existants si tu en as

    repError, camMatrix, distCoeff, rvecs, tvecs = cv2.calibrateCamera(worldPtsList, imgPtsList, (1920,1080), None, None)

    for i, tvec in enumerate(tvecs):
        print(f"Image {i}: distance Z estimée = {tvec[2][0]:.1f} (unité = celle de ton square_size)")

    print("Camera Matrix:\n", camMatrix)
    print("Reproj Error (pixels): {:.4f}".format(repError))

    # Save Calibration
    param_path = os.path.join("/home/skouff/master_thesis/camera_calibration", "calibration_plexi.npz")
    np.savez(param_path, repError=repError, camMatrix=camMatrix, distCoeff=distCoeff, rvecs=rvecs, tvecs=tvecs)

if __name__ == "__main__":
    calibrate()