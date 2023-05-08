import copy
import numpy as np
import cv2
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import pandas as pd
import open3d as o3d
from sklearn.cluster import DBSCAN
from collections import Counter

"""
This file contrains the algorithm and the functions to register the 3D point clouds
based on the disparity maps and the camera parameters. The algorithm is the following:
    1. Load the disparity map and the camera parameters
    2. load the bounding boxes predicted by the model and theirs confidence
    3. For each bounding box in each frame:
        3.1. Get the disparity map of the bounding box applying a mask
        3.2. Get the 3D points based on the disparity map and the camera parameters
        3.3  Post processing of the points to get the points that belong to the object detected
            3.3.1 remove outliers
            3.3.2 Apply custering (DBSCAN)
        
    4. Using the ID from DeepSORT, match the clusters of points beetween frames
    5. Calculate the transformation matrix between the clusters of points (avg. of the points / ICP)

"""


def draw_labels_on_model(pcl, labels):
    """ Recives a point cloud and a list of labels and plot the point cloud with the labels as colors"""
    cmap = plt.get_cmap("tab20")
    pcl_temp = copy.deepcopy(pcl)
    max_label = labels.max()
    print("%s has %d clusters" % (pcl, max_label + 1))
    colors = cmap(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcl_temp.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcl_temp])

def display_inlier_outlier(cloud, ind):
    """ Recives a point cloud and a list of indices and plot the inliers and outliers of the point cloud"""
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])

def pointclouds_from_masks(disparity_frame_:np.ndarray, img_:np.ndarray, obj_masks_:List[np.ndarray], Q:np.ndarray, min_pcd_size_:int) -> List[o3d.geometry.PointCloud]:
    # Get list of objects from both frames
    _objs_pointclouds = make_pointclouds_from_masks(disparity_frame_, img_,  obj_masks_, Q)

    # clustering for eliminating outliers
    post_cluster1_list = []
    for _pcd in _objs_pointclouds:
        # Cluster the point to remove the noise from the background  
        labels = cluster_DBscan(_pcd, eps=0.0050, min_samples=100)
        
        # Get the biggest cluster 
        _pcd = get_biggest_cluster(_pcd, labels)
    
        # TODO [By:Alex]: this is not relevant anymore
        _pcd = remove_outliers_from_pointCloud(_pcd)

        # Calculate the vector of translation
        if len(_pcd.points) > min_pcd_size_:
            post_cluster1_list.append(_pcd)
        
    return post_cluster1_list

def New_ObtainListOfPontClouds(disparity_frame_, n_frame_, left_img_, bb_boxes, Q, extr_mat):

    # Get list of objects from both frames
    bb1 =  bb_boxes[bb_boxes['frame'] == n_frame_]
    cluster_list1 = make_pointclouds_from_masks(disparity_frame_, left_img_,  bb1, Q, extr_mat)

    # plot the point clouds
    #o3d.visualization.draw_geometries(cluster_list1)
    #o3d.visualization.draw_geometries(cluster_list2)
    

    post_cluster1_list = []
    # Get the clusters that are in both frames
    # TODO: match the clusters using the ID from DeepSORT
    for i in range(len(cluster_list1)):

        # get clusters with same ID
        cluster1 = cluster_list1[i]


        # Cluster the point to remove the noise from the background  
        labels1 = cluster_DBscan(cluster1, eps=0.01, min_samples=100)

        # remove outliers stadistical approach
        #draw_labels_on_model(cluster1, labels1)
        #draw_labels_on_model(cluster2, labels2)
        
        # Get the biggest cluster 
        cluster1 = get_biggest_cluster(cluster1, labels1)
    
        # TODO: Alex - this is not relevant anymore
        cluster1 = remove_outliers_from_pointCloud(cluster1)

        # Calculate the vector of translation
        post_cluster1_list.append(cluster1)

        

    # The clusters list should be in order with the IDs from DeepSORT
    # so the index i in the list correspond to the same object in both frames and with the translation vector
    # in the list
    return post_cluster1_list

def make_pointclouds_from_masks(disparity_map, img_, obj_masks_:List[np.ndarray], Q) -> List[o3d.geometry.PointCloud]:
    """
        Extract the list of the pointClouds of the objects detected in the frame.
        NOTE: we are using the bounding box therefore the pointclouds may contains point from the background
        parameters:
            disparity_map: disparity map of the frame
            img_: color image of the frame

            
       
        returns: list of point clouds"""
    pc_from_objmasks = []
    int2color = lambda i: (i*10, i*10, i*10)
    for _i, _mask in enumerate(obj_masks_):
        # apply mask
        maskedDisparity = disparity_map.copy()
        maskedDisparity[_mask == 0] = 0

        # get point cloud
        _mask_points, _mask_points_colors = generate_pointCloud(maskedDisparity, img_, Q)        

        # convert to open3d point cloud and add the color from the img
        _mask_pointcloud = o3d.geometry.PointCloud()
        _mask_pointcloud.points = o3d.utility.Vector3dVector(_mask_points)
        _mask_pointcloud.colors = o3d.utility.Vector3dVector(_mask_points_colors / 255.0)

        # plot the point cloud
        #o3d.visualization.draw_geometries([pcd])

        pc_from_objmasks.append(_mask_pointcloud)

    return pc_from_objmasks

def generate_pointCloud(disparity_map, color_img,  Q):
    """ Function to generate the pointcloud from the disparity map and the Q matrix """
    points = cv2.reprojectImageTo3D(disparity_map, Q, handleMissingValues=False)
    # reflect on x axis 
    reflect_matrix = np.identity(3)
    reflect_matrix[0] *= -1
    #points = np.dot(points, reflect_matrix)
    
    
    # extract colors from the image
    # colors = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB) 
    colors = color_img

    # Remove the points with value 0 (no depth) one applied by mask
    mask = disparity_map > disparity_map.min()
    out_points = points[mask]
    out_colors = colors[mask]
    
    # filter by dimension
    idx = np.fabs(out_points[:,0]) < 4.5
    out_points = out_points[idx]
    out_colors = out_colors.reshape(-1, 3)
    out_colors = out_colors[idx]

    return out_points, out_colors

def remove_outliers_from_pointCloud(pointCloud):
    """Remove outliers using stadistical approach"""
    cl, ind = pointCloud.remove_statistical_outlier(nb_neighbors=2, std_ratio=1.0)
    inlier_cloud = pointCloud.select_by_index(ind)
    return inlier_cloud

def cluster_DBscan(point_cloud, min_samples=50, eps=0.2):
    """Cluster the point cloud using DBSCAN to divide front from back"""
    xyz = np.asarray(point_cloud.points)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(xyz)
    labels = db.labels_
    return labels

def calculate_translation_AvgPoint(pointCloud_frame1, pointCloud_frame2):
    """Calculate the translation between the two point clouds using the average point method
       This fucntion asumes that the pointcloud belong to the same cluster in different frames"""
    avg1 = get_avg_point_pointCloud(pointCloud_frame1)
    avg2 = get_avg_point_pointCloud(pointCloud_frame2)
    return avg2 - avg1

def get_avg_point_pointCloud(pointCloud):
    """ Return the average point of the point cloud"""
    xyz = np.asarray(pointCloud.points)
    return np.mean(xyz, axis=0)

def calculate_translation_ICP(pointCloud_frame1, pointCloud_frame2):
    """Calculate the translation between the two point clouds using the ICP method
       This fucntion asumes that the pointcloud belong to the same cluster in different frames"""
    raise NotImplementedError()

def calculate_bounding_box(pointCloud):
    """Calculate the bounding box of the point cloud"""
    return pointCloud.get_axis_aligned_bounding_box()

def calculate_vector_beetween_cluster(cluster1, cluster2):
    """ Calculate the vector between two clusters
        used to get the vector beetween frames """
    avg1 = get_avg_point_pointCloud(cluster1)
    avg2 = get_avg_point_pointCloud(cluster2)
    return avg2 - avg1

def calculate_rotation_beetween_cluster(cluster1, cluster2):
    """ Calcultate rotation beetween two clusters in y"""
    avg1 = get_avg_point_pointCloud(cluster1)
    avg2 = get_avg_point_pointCloud(cluster2)
    diff = avg2 - avg1


    # get the rotation in y axis
    return diff[1]

def get_biggest_cluster(pointClouds, labels):
    """Get the biggest cluster from the pointclouds"""
   
    id_clusters = Counter(labels).most_common(1)
    id_clusters = [id[0] for id in id_clusters]

    new_cluster = []
    for i, l in enumerate(labels):
        if l == id_clusters:
            new_cluster.append(i)

    # Get the points of the biggest cluster
    newCluster = pointClouds.select_by_index(new_cluster)
    return newCluster

##############################################################################################################
# funtions from the exercises
##############################################################################################################


def get_labels_temp(seq_dir_):
    """ Funtion to load the gt """
    print("TEMPORAL FUNTION WE USE THE ONE THAT IONY DID")
    print("TEMPORAL FUNTION WE USE THE ONE THAT IONY DID")
    print("TEMPORAL FUNTION WE USE THE ONE THAT IONY DID")

    _labels_file = str(seq_dir_ / "labels.txt")
    headers = ['frame', 'track_id', 'type', 'truncated', 'occluded', 'alpha', 'bbox_left', 'bbox_top', 'bbox_right', 'bbox_bottom', 'height', 'width', 'length', 'x', 'y', 'z', 'yaw']
    return pd.read_csv(_labels_file, sep=' ', header=None, names=headers)


def write_ply(fn, verts, colors):
        ply_header = '''ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
        '''
        colors = colors.copy()
        verts = verts.reshape(-1, 3)
        verts = np.hstack([verts, colors])
        with open(fn, 'wb') as f:
            f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
            np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')



if __name__ == "__main__":
    pass
