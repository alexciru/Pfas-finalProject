import cv2
import os
import glob
import numpy as np
from pathlib import Path

""" This file contains the methods used for the depth estimation of the images.
    The methods are:
        - stereoBMMap
        - semiGlobalMatchMap
     Also contains method for post-processing the disparity map to increase the accuracy of the depth estimation.
"""


def filter_disparity_map(sgbmObject, disparityMap, left_img_blur, right_img_blur):
    # Calculate the disparity map and apply WLS filter
    leftMatcher = sgbmObject
    rightMatcher = cv2.ximgproc.createRightMatcher(leftMatcher)

    # Calculating disparity using the stereoBM algorithm
    leftDisparity = leftMatcher.compute(left_img_blur, right_img_blur)
    rightDisparity = rightMatcher.compute(right_img_blur, left_img_blur)

    # Create a WLS (weighted least squares) filter (source: https://docs.opencv.org/3.4/d3/d14/tutorial_ximgproc_disparity_filtering.html)
    wlsFilter = cv2.ximgproc.createDisparityWLSFilter(leftMatcher)
    wlsFilter.setLambda(
        8000
    )  # The tuning parameter, depends on the range of disparity values
    wlsFilter.setSigmaColor(
        0.7
    )  # Adjusts the filter's sensitivity to edges in the image (between 0.8 and 2.0 usually good)

    filteredDisparity = wlsFilter.filter(
        leftDisparity, left_img_blur, None, rightDisparity
    )

    # Get the original and filtered disparity images
    orgDistL = cv2.ximgproc.getDisparityVis(leftDisparity)
    filteredL = cv2.ximgproc.getDisparityVis(filteredDisparity)

    return filteredDisparity, filteredL


def semiGlobalMatchMap(left_img, right_img):
    """Function using the SGMBM algorithm to compute the disparity map

    Args:
        left_img (ndarray): color image of the left camera
        right_img (ndarray): color image of the right camera

    Returns:
        stereo: the SGMBM object
        disparity: the disparity map
    """
    ###### Default values ######
    blockSize = 7  # odd number, usually in range 3-11
    minDisparity = -1
    maxDisparity = 6
    numDisparities = (
        maxDisparity - minDisparity
    )  # max disparity minus minDisparity, must be divisible by 16
    preFilterCap = 1
    uniquenessRatio = 1
    speckleRange = 1  # multiplied by 16 implicitly, 1 or 2 usually good
    speckleWindowSize = 54  # 50-200 range
    ###################

    # Creating an object of StereoBM algorithm
    # Updating the parameters based on the trackbar positions
    numDisparities = numDisparities * 16
    if blockSize % 2 == 0:
        blockSize += 1

    if blockSize < 5:
        blockSize = 5

    leftMatcher = cv2.StereoSGBM_create()

    # Setting the updated parameters before computing disparity map
    leftMatcher.setNumDisparities(numDisparities)
    leftMatcher.setBlockSize(blockSize)
    leftMatcher.setUniquenessRatio(uniquenessRatio)
    leftMatcher.setSpeckleRange(speckleRange)
    leftMatcher.setSpeckleWindowSize(speckleWindowSize)
    leftMatcher.setMinDisparity(minDisparity)

    # P1 and P2 values from OpenCV documentation
    leftMatcher.setP1(8 * 3 * blockSize**2)
    leftMatcher.setP2(32 * 3 * blockSize**2)

    # Blur the images to increase the accuracy of the disparity map

    left_img_blur = cv2.blur(cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY), (5, 5))
    right_img_blur = cv2.blur(cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY), (5, 5))

    # Calculating disparity using the stereoBM algorithm
    leftDisparity = leftMatcher.compute(left_img_blur, right_img_blur).astype(
        np.float32
    )
    distL = cv2.ximgproc.getDisparityVis(leftDisparity)

    filteredDisparity, filteredL = filter_disparity_map(
        leftMatcher, distL, left_img_blur, right_img_blur
    )

    return filteredDisparity, filteredL


def get_cam_matrices(calib_filepath_: Path):
    """Function to read all the interesting matrices in the calib_cam_to_cam.txt file

    Returns:
        matrices: all the relevant matrices for the colored images
    """

    with open(calib_filepath_, "r") as f:
        fin = f.readlines()
        for line in fin:
            if line[:4] == "R_02":
                rotation2 = (
                    np.array(line[6:].strip().split(" "))
                    .astype("float32")
                    .reshape(3, -1)
                )
            elif line[:4] == "R_03":
                rotation3 = (
                    np.array(line[6:].strip().split(" "))
                    .astype("float32")
                    .reshape(3, -1)
                )
            elif line[:4] == "K_02":
                intrinsic2 = (
                    np.array(line[6:].strip().split(" "))
                    .astype("float32")
                    .reshape(3, -1)
                )
            elif line[:4] == "K_03":
                intrinsic3 = (
                    np.array(line[6:].strip().split(" "))
                    .astype("float32")
                    .reshape(3, -1)
                )
            elif line[:4] == "T_02":
                translation2 = (
                    np.array(line[6:].strip().split(" "))
                    .astype("float32")
                    .reshape(3, -1)
                )
            elif line[:4] == "T_03":
                translation3 = (
                    np.array(line[6:].strip().split(" "))
                    .astype("float32")
                    .reshape(3, -1)
                )
            elif line[:9] == "S_rect_02":
                imageSize2 = np.array(line[11:].strip().split(" ")).astype("float32")
            elif line[:9] == "S_rect_03":
                imageSize3 = np.array(line[11:].strip().split(" ")).astype("float32")
            elif line[:9] == "R_rect_02":
                rectRot2 = (
                    np.array(line[11:].strip().split(" "))
                    .astype("float32")
                    .reshape(3, -1)
                )
            elif line[:9] == "R_rect_03":
                rectRot3 = (
                    np.array(line[11:].strip().split(" "))
                    .astype("float32")
                    .reshape(3, -1)
                )
            elif line[:9] == "P_rect_02":
                camMatrix2 = (
                    np.array(line[11:].strip().split(" "))
                    .astype("float32")
                    .reshape(3, -1)
                )
            elif line[:9] == "P_rect_03":
                camMatrix3 = (
                    np.array(line[11:].strip().split(" "))
                    .astype("float32")
                    .reshape(3, -1)
                )

    return (
        rotation2,
        rotation3,
        translation2,
        translation3,
        imageSize2,
        imageSize3,
        rectRot2,
        rectRot3,
        camMatrix2,
        camMatrix3,
        intrinsic2,
        intrinsic3,
    )


def get_Q_matrix(img_size: tuple, calib_filepath_: Path):
    _, _, _, _, _, _, _, _, projection_mat_l, projection_mat_r, _, _ = get_cam_matrices(
        calib_filepath_
    )

    projection_mat_l = projection_mat_l[:, :3]
    projection_mat_r = projection_mat_r[:, :3]
    Tmat = np.array([0.54, 0.0, 0.0])  # From the KITTI Sensor setup, in metres

    cvQ = np.zeros((4, 4))
    cv2.stereoRectify(
        cameraMatrix1=projection_mat_l,
        cameraMatrix2=projection_mat_r,
        distCoeffs1=0,
        distCoeffs2=0,
        imageSize=img_size,
        R=np.identity(3),
        T=Tmat,
        R1=None,
        R2=None,
        P1=None,
        P2=None,
        Q=cvQ,
    )
    return cvQ


def test_module():
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    DATA_DIR = ROOT_DIR + "\\data\\final_project_2023_rect"
    SEQ_01 = DATA_DIR + "\\seq_01"

    n_frame1 = 0
    seq = SEQ_01
    ###########################################

    ## Get the images
    left_path = SEQ_01 + "\\image_02\\data\\*.png"
    right_path = SEQ_01 + "\\image_03\\data\\*.png"

    left_images = glob.glob(left_path)
    right_images = glob.glob(right_path)
    left_images.sort()
    right_images.sort()

    left_img1 = cv2.imread(left_images[n_frame1])
    right_img1 = cv2.imread(right_images[n_frame1])
    disparity, __ = semiGlobalMatchMap(left_img1, right_img1)


if __name__ == "__main__":
    print("Running test module in DepthStimation.py")
    test_module()
