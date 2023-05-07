import numpy as np
import os
import cv2
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
import open3d as o3d
from tqdm import tqdm
from Depth.DephtEstimation import semiGlobalMatchMap, get_Q_matrix, readAllColorMatrices
from final_project.src.depth.registration import get_labels_temp, ObtainListOfPontClouds
from ResultSaving import write_results_to_file

from utils.utils import get_frames, SEQ_01

def main():
    print("\n--- Starting 3D reconstruction pipeline ---")
        
    ################# VALUES ###################
    _n_frames = 2
    # frames = [(_l, _r) for _l, _r in [get_frames(frame_num_=i, seq_dir_=SEQ_01) for i in range(_n_frames)]]
    ###########################################


    # for _frame_t, (_frame_l_t, _frame_r_t) in enumerate(frames):
    #     # 1. get disparity map
    #     ...

    # Function to get the labels
    bb_boxes = get_labels_temp(SEQ_01)
    ## All the frames 
    for n_frame in tqdm(range(2)):
        
        # left_img1 = cv2.imread(left_images[n_frame])
        # right_img1 = cv2.imread(right_images[n_frame])
        left_img1, right_img1 = get_frames(n_frame, SEQ_01)
        left_img2, right_img2 = get_frames(n_frame+1, SEQ_01)
    

        # n_frame2 = n_frame + 1
        # left_img2 = cv2.imread(left_images[n_frame2])
        # right_img2 = cv2.imread(right_images[n_frame2])
        
        Q = get_Q_matrix(left_img1.shape[:2])
        ex_mat = np.array([[9.999838e-01, -5.012736e-03, -2.710741e-03, 5.989688e-02],
                            [5.002007e-03, 9.999797e-01, -3.950381e-03, -1.367835e-03],
                            [2.730489e-03, 3.936758e-03, 9.999885e-01, 4.637624e-03],
                            [0,0,0,1]])


        # get the filtered version of the disparity map
        disparity_frame1, __ = semiGlobalMatchMap(left_img1, right_img1)
        disparity_frame2, __ = semiGlobalMatchMap(left_img2, right_img2)


        # Finnally the pointclouds
        clusterlist1 = ObtainListOfPontClouds(disparity_frame1, n_frame, left_img1, bb_boxes, Q, ex_mat)

        # TODO: no need to compute the clusters again if was computer in the previous frame
        clusterlist2 = ObtainListOfPontClouds(disparity_frame2, n_frame+1, left_img2, bb_boxes, Q, ex_mat)
        
        
        # Plot the resutls
        o3d.visualization.draw_geometries(clusterlist1)
        # o3d.visualization.draw_geometries(cluster2_list)

        # write_results_to_file(n_frame, None, clusterlist1, clusterlist2, filename = "results.txt")


if __name__ == "__main__":
    main()