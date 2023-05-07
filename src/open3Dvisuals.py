import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import copy
import os 
import glob 
import pandas as pd
import cv2
from sklearn.cluster import DBSCAN
from collections import Counter

def readPrects():
    """Function to read all the interesting matrices in the calib_cam_to_cam.txt file.
       File should be inside data/final_project_2023_rect

    Returns:
        matrices: all the relevant matrices for the colored images
    """    
    
    path = "../data/final_project_2023_rect/calib_cam_to_cam.txt"
    
    with open(path, 'r') as f:
        fin = f.readlines()
        for line in fin:
            if line[:9] == "P_rect_02":
                p2 = np.array(line[11:].strip().split(" ")).astype('float32').reshape(3,-1)
            elif line[:9] == "P_rect_03":
                p3 = np.array(line[11:].strip().split(" ")).astype('float32').reshape(3,-1)
                break
    
    return p2, p3

def get_labels_temp(seq_dir_):
    """ Funtion to load the gt """
    _labels_file = str(seq_dir_ + "/labels.txt")
    headers = ['frame', 'track_id', 'type', 'truncated', 'occluded', 'alpha', 'bbox_left', 'bbox_top', 'bbox_right', 'bbox_bottom', 'height', 'width', 'length', 'x', 'y', 'z', 'yaw']
    return pd.read_csv(_labels_file, sep=' ', header=None, names=headers)

def get_Q_matrix(img_size):
    cam2, cam3 = readPrects()
    cam2 = cam2[:,:3]
    cam3 = cam3[:,:3]
    Tmat = np.array([0.54, 0.0, 0.0])   # From the KITTI Sensor setup, in metres 
    cvQ = np.zeros((4,4))
    cv2.stereoRectify(cameraMatrix1=cam2, cameraMatrix2=cam3, distCoeffs1=0, distCoeffs2=0,
                        imageSize=img_size, R=np.identity(3), T=Tmat, 
                        R1=None, R2=None,P1=None, P2=None, Q=cvQ)

    return cvQ

def getDepthMap(imgL, imgR, sgbm, view=False):
    """Function to get the depth map from the left and right images.
    Args:
        imgL: left image
        imgR: right image
        sgbm: the stereoBM object
    Returns:
        disp: the disparity map
    """    
    left_img_blur = cv2.blur(cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY), (5,5))
    right_img_blur = cv2.blur(cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY), (5,5))
    # Calculate the disparity map and apply WLS filter
    leftMatcher = sgbm

    # Create a WLS (weighted least squares) filter (source: https://docs.opencv.org/3.4/d3/d14/tutorial_ximgproc_disparity_filtering.html) 
    wlsFilter = cv2.ximgproc.createDisparityWLSFilter(leftMatcher)
    wlsFilter.setLambda(8000)     # The tuning parameter, depends on the range of disparity values
    wlsFilter.setSigmaColor(0.7)    # Adjusts the filter's sensitivity to edges in the image (between 0.8 and 2.0 usually good)
    rightMatcher = cv2.ximgproc.createRightMatcher(leftMatcher)

    # Calculating disparity using the stereoBM algorithm
    leftDisparity =  leftMatcher.compute(left_img_blur, right_img_blur)
    rightDisparity = rightMatcher.compute(right_img_blur, left_img_blur)

    filteredDisparity = wlsFilter.filter(leftDisparity, left_img_blur, None, rightDisparity)

    # Get the original and filtered disparity images
    orgDistL = cv2.ximgproc.getDisparityVis(leftDisparity)
    filteredL = cv2.ximgproc.getDisparityVis(filteredDisparity)
    
    if view:
        # Concatenate the two disparity maps before viewing
        concatDisp = np.concatenate((orgDistL, filteredL), axis=0)
        cv2.imshow("Disparity[top:org, bottom:filtered]", concatDisp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return filteredDisparity

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
        colors = colors.copy().reshape(-1, 3)
        verts = verts.reshape(-1, 3)
        verts = np.hstack([verts, colors])
        with open(fn, 'wb') as f:
            f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
            np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


def generate_pointCloud(disparity_map,color_img,  Q):
        """ Function to generate the point cloud from the disparity map and the Q matrix """
        points = cv2.reprojectImageTo3D(disparity_map, Q, handleMissingValues=False)
        # reflect on x axis 
        reflect_matrix = np.identity(3)
        reflect_matrix[0] *= -1
        # points = np.matmul(points, reflect_matrix)
        
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
        
        # write_ply('out.ply', out_points, out_colors)

        return out_points, out_colors

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
        # cv2.imshow("masked disparity", maskedDisparity)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        # get point cloud
        points , colors = generate_pointCloud(maskedDisparity, color_img, Q)
        if points.shape[0] < 5000:
            # Not enough points to be considered an object 
            continue

        # transform pointcloud by the extrinsic matrix to get the pointcloud in the world coordinates
        

        # convert to open3d point cloud and add the color from the img
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors / 255) # Remember to divide by 255 to get the
                                                              # colors right! Needs to be 0-1 range.

        # plot the point cloud
        # o3d.visualization.draw_geometries([pcd])
        # TODO: store openCV cluster
        clusters.append(pcd)

    return clusters

def cluster_DBscan(point_cloud, min_samples=50, eps=0.2):
    """Cluster the point cloud using DBSCAN to divide front from back"""
    xyz = np.asarray(point_cloud.points)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(xyz)
    labels = db.labels_
    return labels

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

def get_avg_point_pointCloud(pointCloud):
    """ Return the average point of the point cloud"""
    xyz = np.asarray(pointCloud.points)
    return np.mean(xyz, axis=0)

def calculate_translation_AvgPoint(pointCloud_frame1, pointCloud_frame2):
    """Calculate the translation between the two point clouds using the average point method
       This fucntion asumes that the pointcloud belong to the same cluster in different frames"""
    avg1 = get_avg_point_pointCloud(pointCloud_frame1)
    avg2 = get_avg_point_pointCloud(pointCloud_frame2)
    return avg2 - avg1


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

        # Cluster the point to remove the noise from the background
  
        labels1 = cluster_DBscan(cluster1, eps=0.01, min_samples=100)
        labels2 = cluster_DBscan(cluster2, eps=0.01, min_samples=100)

        #draw_labels_on_model(cluster1, labels1)
        #draw_labels_on_model(cluster2, labels2)
        

        # Get the biggest cluster 
        cluster1 = get_biggest_cluster(cluster1, labels1)
        cluster2 = get_biggest_cluster(cluster2, labels2)

        
        # o3d.visualization.draw_geometries([cluster1])
        # o3d.visualization.draw_geometries([cluster2])
        # TODO: Alex - this is not relevant anymore
        # cluster1 = remove_outliers_from_pointCloud(cluster1)
        # cluster2 = remove_outliers_from_pointCloud(cluster2)
        

        # Calculate the vector of translation
        post_cluster1_list.append(cluster1)
        post_cluster2_list.append(cluster2)
        
        translation = calculate_translation_AvgPoint(cluster1, cluster2)
        translation_list.append(translation)

    # The clusters list should be in order with the IDs from DeepSORT
    # so the index i in the list correspond to the same object in both frames and with the translation vector
    # in the list
    return post_cluster1_list, post_cluster2_list, translation_list


def calculate_bounding_box(pointCloud, color = (0,255,0)):
    """Calculate the bounding box of the point cloud"""
    return pointCloud.get_axis_aligned_bounding_box()

def calculate_rotation_beetween_cluster(cluster1, cluster2):
    """ Calcultate rotation beetween two clusters in y"""
    avg1 = get_avg_point_pointCloud(cluster1)
    avg2 = get_avg_point_pointCloud(cluster2)
    diff = avg2 - avg1
    # get the rotation in y axis
    return diff[1]

def get_avg_point_pointCloud(pointCloud):
    """ Return the average point of the point cloud"""
    xyz = np.asarray(pointCloud.points)
    return np.mean(xyz, axis=0)


def write_results_to_file(frame_id, DeepSortId, clusterlist1_list, cluster2_list, img, filename = "results.txt"):
    """
    Funtion to save the results of the object detection to a file
    Format:
        frame_id, track_id, type, truncated, occluded, alpha, bbox_left, bbox_top, bbox_right, bbox_bottom, height, width, length, x, y, z, rotation_y
    """
    
    # Get the data from the pointcloud1 
    # pointcloud2 only use for calculating the vector beetween frames
    # and the translation vector
    

    frame = frame_id
    for i in range(len(clusterlist1_list)):
        row = []

        cluster = clusterlist1_list[i]
        # mirrorBack = np.matmul(cluster.points, np.asarray([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        # cluster.points = o3d.utility.Vector3dVector(mirrorBack)

        track_id = 1 # TODO:  change this
        otype = "Car"# TODO:  change this
        truncated = 0 # TODO:  change this
        occluded = 0 # TODO:  change this
        alpha = 0 # TODO:  change this

        # 2D bounding box of object in the image
        # Obtain from YoloInDeepSort
        bbox_left = 0 
        bbox_top = 0 
        bbox_right = 0 
        bbox_bottom = 0 

        bbox = calculate_bounding_box(cluster)
        
        max_bound = bbox.get_max_bound()
        min_bound = bbox.get_min_bound()

        # Calculate the dimensions
        width = max_bound[0] - min_bound[0]
        height = max_bound[1] - min_bound[1]
        length = max_bound[2] - min_bound[2]
        # dimensions in camera coordinates

        # center location in camera coordinates
        # obtain from cluter, make a box and get the center
        # avg_point = get_avg_point_pointCloud(cluster)
        box_corners = np.asarray(bbox.get_box_points())
        ################# Corner positions ##########
        # box_corners[0] = behind, down and left
        # box_corners[1] = behind, down and right
        # box_corners[2] = behind, up and left
        # box_corners[3] = in front, down and left
        # box_corners[4] = in front, up and right
        # box_corners[5] = in front, up and left
        # box_corners[6] = in front, down and right
        # box_corners[7] = behind, up and right
        #############################################
        
        lower_corners = np.asarray([box_corners[0], box_corners[1], 
                                    box_corners[3], box_corners[6]])
        
        # Find the center of the lower corners
        lower_center = np.mean(lower_corners, axis=0)

        # create bbox from cluster
        # convert to left hand coordinates for openCV
        x = lower_center[0]
        y = lower_center[1]
        z = lower_center[2]

        # Display the point cloud with open3d 
        centerBox = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        centerBox.paint_uniform_color([1, 0, 0]) # assign color
        centerBox.translate(lower_center)
        # o3d.visualization.draw_geometries([cluster, centerBox]) # Uncomment to display the point cloud
                                                                  # and the center of the lower corners

        # Obtain from clusterList2 with same index and obtain rotation of vector
        rotation_y = calculate_rotation_beetween_cluster(cluster, cluster2_list[i])
        
        # Obtain from DeepSort
        score = 69.420 

        row.append(frame)
        row.append(track_id)
        row.append(otype)
        row.append(truncated)
        row.append(occluded)
        row.append(alpha)
        row.append(bbox_left)
        row.append(bbox_top)
        row.append(bbox_right)
        row.append(bbox_bottom)
        row.append(height)
        row.append(width)
        row.append(length)
        row.append(x)
        row.append(y)
        row.append(z)
        row.append(rotation_y)
        row.append(score)

        
        formatted_data = ' '.join(str(value) for value in row) + '\n'

        # Write the formatted data to file
        with open(filename, 'a') as f:
            f.write(formatted_data)




########### EXECUTION HERE ##############

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = ROOT_DIR + "/data/final_project_2023_rect"
SEQ_01 = DATA_DIR + "/seq_01"
SEQ_02 = DATA_DIR + "/seq_02"
SEQ_03 = DATA_DIR + "/seq_03"

## Get the images
left_path = SEQ_01 + "/image_02/data/*.png"
right_path = SEQ_01 + "/image_03/data/*.png"

left_images = glob.glob(left_path)
right_images = glob.glob(right_path)
left_images.sort()
right_images.sort()

# Function to get the labels
bb_boxes = get_labels_temp(SEQ_01)

# Create the sgbm object with known good values
numDisp, minDisp, blockS, unique, speckR, speckWindow = 6*16, 5, 7, 1, 2, 224
P1, P2 = 8*3*blockS**2, 32*2*blockS**2

sgbm = cv2.StereoSGBM_create(numDisparities=numDisp, minDisparity=minDisp, blockSize=blockS,
                                   uniquenessRatio=unique, speckleRange=speckR, 
                                   speckleWindowSize=speckWindow, 
                                   P1=8*3*blockS**2, P2=32*3*blockS**2)

## All the frames 
Q = get_Q_matrix(cv2.imread(left_images[0]).shape[:2]) # Calculate the disparity-to-depth mapping matrix (Q)
for n_frame in range(len(left_images)-1):    
    left_img1 = cv2.imread(left_images[n_frame])
    right_img1 = cv2.imread(right_images[n_frame])

    n_frame2 = n_frame + 1 # Get the next frame
    left_img2 = cv2.imread(left_images[n_frame2])
    right_img2 = cv2.imread(right_images[n_frame2])
    
    # get the filtered version of the disparity map
    disparity_frame1 = getDepthMap(left_img1, right_img1, sgbm, view=False)
    disparity_frame2 = getDepthMap(left_img2, right_img2, sgbm, view=False)
    
    # Finnally the pointclouds
    clusterlist1, clusterlist2, translation = ObtainListOfPontClouds(disparity_frame1
                                                            ,n_frame, disparity_frame2, n_frame2,left_img1, left_img2, bb_boxes, Q)
    
    write_results_to_file(n_frame, None, clusterlist1, clusterlist2, left_img1, filename="results-testing.txt")
        
        
        