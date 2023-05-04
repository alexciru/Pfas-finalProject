import numpy as np
import os
import cv2
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
import open3d as o3d

from DephtStimation import semiGlobalMatchMap, get_Q_matrix, readAllColorMatrices
from registration import get_labels_temp, ObtainListOfPontClouds
from ResultSaving import write_results_to_file


def main():

    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    DATA_DIR = ROOT_DIR + "\\data\\final_project_2023_rect"
    SEQ_01 = DATA_DIR + "\\seq_01"
    SEQ_02 = DATA_DIR + "\\seq_01"
    SEQ_03 = DATA_DIR + "\\seq_01"

    ############  values   ####################
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
   

    n_frame2 = n_frame1 + 1
    left_img2 = cv2.imread(left_images[n_frame2])
    right_img2 = cv2.imread(right_images[n_frame2])


    bb_boxes = get_labels_temp(SEQ_01)
    Q = get_Q_matrix(left_img1.shape[:2])
    ex_mat = np.array([[9.999838e-01, -5.012736e-03, -2.710741e-03, 5.989688e-02],
                        [5.002007e-03, 9.999797e-01, -3.950381e-03, -1.367835e-03],
                        [2.730489e-03, 3.936758e-03, 9.999885e-01, 4.637624e-03],
                        [0,0,0,1]])


    # get the filtered version of the disparity map
    disparity_frame1, __ = semiGlobalMatchMap(left_img1, right_img1)
    disparity_frame2, __ = semiGlobalMatchMap(left_img2, right_img2)


    # Finnally the pointclouds
    clusterlist1, cluster2_list, translation = ObtainListOfPontClouds(disparity_frame1
                                                            ,n_frame1, disparity_frame2, n_frame2,left_img1, left_img2, bb_boxes, Q, ex_mat)
    
    o3d.visualization.draw_geometries([clusterlist1[0], clusterlist1[1]] )
    o3d.visualization.draw_geometries(cluster2_list)

    
    write_results_to_file(n_frame1, None, clusterlist1, cluster2_list, filename = "results.txt")




if __name__ == "__main__":
    main()