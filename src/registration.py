import numpy as np
import os
import cv2
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
import open3d as o3d
from sklearn.cluster import KMeans, k_means, DBSCAN
import copy
from collections import Counter
from DephtStimation import semiGlobalMatchMap, readAllColorMatrices, get_Q_matrix

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


def ObtainListOfPontClouds(disparity_frame1,n_frame1, disparity_frame2, n_frame2,left_img1, left_img2, bb_boxes, Q):
    """    
    This is the main function of the algorithm. It takes the disparity maps of two frames and the bounding boxes
    parameters:
        disparity_frame1: disparity map of the first frame
        n_frame1: number of the first frame
        disparity_frame2: disparity map of the second frame
        n_frame2: number of the second frame
        bb_boxes: pandas dataframe with the bounding boxes of the two frames AND THE ID FROM DEEPSORT
        Q: Q matrix from the camera parameters

    returns: list of point clouds of the objects detected in the two frames with their transformation matrix
    """
    # Get list of objects from both frames
    bb1 =  bb_boxes[bb_boxes['frame'] == n_frame1]
    cluster_list1 = extract_objects_point_clouds(disparity_frame1, left_img1,  bb1, Q)
    bb2 =  bb_boxes[bb_boxes['frame'] == n_frame2]
    cluster_list2 = extract_objects_point_clouds(disparity_frame2, left_img2,  bb2, Q)

    # plot the point clouds
    # o3d.visualization.draw_geometries(cluster_list1)
    # o3d.visualization.draw_geometries(cluster_list2)
    
    # Get the clusters that are in both frames
    n_matches = min(len(cluster_list1), len(cluster_list2))

    post_cluster1_list = []
    post_cluster2_list = []
    translation_list = []
    # Get the clusters that are in both frames
    # TODO: match the clusters using the ID from DeepSORT
    for i in range(n_matches):

        # get clusters with same ID
        cluster1 = cluster_list1[i]
        cluster2 = cluster_list2[i]

        # Remove outliers

        # remove outliers using standar deviation
        # TODO: Alex - this is not relevant anymore
        # cluster1 = remove_outliers_from_pointCloud(cluster1)
        # cluster2 = remove_outliers_from_pointCloud(cluster2)

        # Cluster the point to remove the noise from the background
        labels1 = cluster_pointCloud(cluster1)
        labels2 = cluster_pointCloud(cluster2)


        # Get the biggest cluster 
        cluster1 = get_biggest_cluster(cluster1, labels1)
        cluster2 = get_biggest_cluster(cluster2, labels2)
    
        # Calculate the vector of translation
        post_cluster1_list.append(cluster1)
        post_cluster2_list.append(cluster2)
        
        translation = calculate_translation_AvgPoint(cluster1, cluster2)
        translation_list.append(translation)

    # The clusters list should be in order with the IDs from DeepSORT
    # so the index i in the list correspond to the same object in both frames and with the translation vector
    # in the list
    return post_cluster1_list, post_cluster2_list, translation_list


def extract_objects_point_clouds(disparity_map, color_img,  bb_detection, Q):
    """Method to extract the list of the pointClouds of the objects detected in the frame
       we are using the bounding box therefore the pointclouds may contains point from the background
       
       returns: list of point clouds"""
    clusters = []
    # TODO: change for new bb_detection one pipeline implemented
    for bb in bb_detection[['bbox_left', 'bbox_top', 'bbox_right', 'bbox_bottom']].values:
        # create mask

        mask = np.zeros(disparity_map.shape, dtype=np.uint8)
        mask[int(bb[1]):int(bb[3]), int(bb[0]): int(bb[2])] = 255

        # apply mask
        maskedDisparity = disparity_map.copy()
        maskedDisparity[mask == 0] = 0

        # get point cloud
        points , colors = generate_pointCloud(maskedDisparity, color_img, Q)
        write_ply("tmp_pointcloud.ply", points, colors)
        pcd = o3d.io.read_point_cloud("tmp_pointcloud.ply")

        # convert to open3d point cloud and add the color from the img
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points)
        # pcd.colors = o3d.utility.Vector3dVector(colors) 

        # plot the point cloud
        #o3d.visualization.draw_geometries([pcd])
        
        clusters.append(pcd)

    return clusters


def generate_pointCloud(disparity_map,color_img,  Q):
        """ Function to generate the point cloud from the disparity map and the Q matrix """
        points = cv2.reprojectImageTo3D(disparity_map, Q, handleMissingValues=False)
        # reflect on x axis 
        reflect_matrix = np.identity(3)
        reflect_matrix[0] *= -1
        points = np.matmul(points,reflect_matrix)
        
        # extract colors from the image
        colors = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB) 

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
    cl, ind = pointCloud.remove_statistical_outlier(nb_neighbors=2, std_ratio=2.0)
    inlier_cloud = pointCloud.select_by_index(ind)
    return inlier_cloud

def cluster_pointCloud(point_cloud, min_samples=50, eps=0.2):
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

def calculate_bounding_box(pointCloud, color = (0,255,0)):
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
    


def test_module():
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
    leftMatcher, distL = semiGlobalMatchMap(left_img1, right_img1)


    n_frame2 = n_frame1 + 1
    left_img2 = cv2.imread(left_images[n_frame2])
    right_img2 = cv2.imread(right_images[n_frame2])

    leftMatcher, distL = semiGlobalMatchMap(left_img2, right_img2)

    bb_boxes = get_labels_temp(SEQ_01)
    Q = get_Q_matrix(left_img1.shape[:2])


    # get the filtered version of the disparity map
    __, disparity_frame1 = semiGlobalMatchMap(left_img1, right_img1)
    __, disparity_frame2 = semiGlobalMatchMap(left_img2, right_img2)


    # Finnally the pointclouds
    clusterlist1, cluster2_list, translation = ObtainListOfPontClouds(disparity_frame1
                                                            ,n_frame1, disparity_frame2, n_frame2,left_img1, left_img2, bb_boxes, Q)
    
    o3d.visualization.draw_geometries(clusterlist1)
    o3d.visualization.draw_geometries(cluster2_list)


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


def export_pointcloud(disparity_map, colors, filename):

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
        with open("pointclouds/"+fn, 'wb') as f:
            f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
            np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')
    
    rot2, rot3, trans2, trans3, imgSize2, imgSize3, rectRot2, rectRot3,\
    cam2, cam3, k2, k3 = readAllColorMatrices()
    # Q = calculateQManually(cam2, cam3)        # We let opencv calculate the Q matrix
    
    cam2 = cam2[:,:3]
    cam3 = cam3[:,:3]
    Tmat = np.array([0.54, 0.0, 0.0])   # From the KITTI Sensor setup, in metres 
    cvQ = np.zeros((4,4))
    cv2.stereoRectify(cameraMatrix1=cam2, cameraMatrix2=cam3, distCoeffs1=0, distCoeffs2=0,
                        imageSize=colors.shape[:2], R=np.identity(3), T=Tmat, 
                        R1=None, R2=None,P1=None, P2=None, Q=cvQ)
    
    points = cv2.reprojectImageTo3D(disparity_map, cvQ, handleMissingValues=False)
    #reflect on x axis
    reflect_matrix = np.identity(3)
    reflect_matrix[0] *= -1
    points = np.matmul(points,reflect_matrix)
    
    colors = cv2.cvtColor(colors, cv2.COLOR_BGR2RGB) # Extract colors from image
    mask = disparity_map > disparity_map.min()
    out_points = points[mask]
    out_colors = colors[mask]
    
    #filter by dimension
    idx = np.fabs(out_points[:,0]) < 4.5
    out_points = out_points[idx]
    out_colors = out_colors.reshape(-1, 3)
    out_colors = out_colors[idx]

    write_ply(filename, out_points, out_colors)
    print(f'{filename} saved')
    return out_points, reflect_matrix, idx, cam3, mask


def get_labels_temp(seq_dir_):
    """ Funtion to load the gt """
    print("TEMPORAL FUNTION WE USE THE ONE THAT IONY DID")
    print("TEMPORAL FUNTION WE USE THE ONE THAT IONY DID")
    print("TEMPORAL FUNTION WE USE THE ONE THAT IONY DID")

    _labels_file = str(seq_dir_ + "\\labels.txt")
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


# small main to test the module

if __name__ == "__main__":
    print("##################################")
    print("# Running the pointcloud module  #")
    print("##################################")
    test_module()
