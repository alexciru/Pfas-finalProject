import numpy as np
import os
import pathlib
import cv2
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
import open3d as o3d
from sklearn.cluster import KMeans, k_means, DBSCAN
import copy

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



def ObtainListOfPontClouds(disparity_frame1,n_frame1, disparity_frame2, n_frame2 ,bb_boxes, Q):
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
    cluster_list1 = extract_objects_point_clouds(disparity_frame1, bb1, Q)
    bb2 =  bb_boxes[bb_boxes['frame'] == n_frame2]
    cluster_list2 = extract_objects_point_clouds(disparity_frame2, bb2, Q)

    # Get the clusters that are in both frames
    n_matches = min(len(cluster_list1, cluster_list2))

    translation_list = []
    # Get the clusters that are in both frames
    # TODO: match the clusters using the ID from DeepSORT
    for i in range(n_matches):

        # get clusters with same ID
        cluster1 = cluster_list1[i]
        cluster2 = cluster_list2[i]

        # Remove outliers
        cluster1 = remove_outliers_from_pointCloud(cluster1)
        cluster2 = remove_outliers_from_pointCloud(cluster2)

        # Cluster the point cloud
        labels1 = cluster_pointCloud(cluster1)
        labels2 = cluster_pointCloud(cluster2)

        # Calculate the vector of translation
        translation = calculate_translation_AvgPoint(cluster1, cluster2)
        translation_list.append(translation)

    # The clusters list should be in order with the IDs from DeepSORT
    # so the index i in the list correspond to the same object in both frames and with the translation vector
    # in the list
    return cluster1, cluster2, translation_list


def extract_objects_point_clouds(disparity_map, bb_detection, Q):
    """Method to extract the list of the pointClouds of the objects detected in the frame
       we are using the bounding box therefore the pointclouds may contains point from the background
       
       returns: list of point clouds"""
    clusters = []
    for bb in bb_detection:
        # create mask
        mask = np.zeros(disparity_map.shape, dtype=np.uint8)
        mask[int(bb[1]):int(bb[3]), int(bb[0]): int(bb[2])] = 255

        # apply mask
        maskedDisparity = disparity_map.copy()
        maskedDisparity[mask == 0] = 0

        # get point cloud
        points = generate_pointCloud(maskedDisparity, Q)
        clusters.append(points)

    return clusters

def generate_pointCloud(disparity_map, Q):
        """ Function to generate the point cloud from the disparity map and the Q matrix """
        points = cv2.reprojectImageTo3D(disparity_map, Q, handleMissingValues=False)

        # reflect on x axis 
        reflect_matrix = np.identity(3)
        reflect_matrix[0] *= -1
        points = np.matmul(points,reflect_matrix)
        
        # extract colors from the image
        colors = cv2.cvtColor(colors, cv2.COLOR_BGR2RGB) 

        # Remove the points with value 0 (no depth) one applied by mask
        mask = disparity_map > disparity_map.min()
        out_points = points[mask]
        out_colors = colors[mask]
        
        # filter by dimension
        idx = np.fabs(out_points[:,0]) < 4.5
        out_points = out_points[idx]
        out_colors = out_colors.reshape(-1, 3)
        out_colors = out_colors[idx]

        return out_points

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

##############################################################################################################
# funtions from the exercises
##############################################################################################################

def readAllColorMatrices():
    """Function to read all the interesting matrices in the calib_cam_to_cam.txt file

    Returns:
        matrices: all the relevant matrices for the colored images
    """    
    path = "../data/final_project_2023_rect/calib_cam_to_cam.txt"
    
    with open(path, 'r') as f:
        fin = f.readlines()
        for line in fin:
            if line[:4] == "R_02":
                rotation2 = np.array(line[6:].strip().split(" ")).astype('float32').reshape(3,-1)
            elif line[:4] == "R_03":
                rotation3 = np.array(line[6:].strip().split(" ")).astype('float32').reshape(3,-1)
            elif line[:4] == "K_02":
                intrinsic2 = np.array(line[6:].strip().split(" ")).astype('float32').reshape(3,-1)
            elif line[:4] == "K_03":
                intrinsic3 = np.array(line[6:].strip().split(" ")).astype('float32').reshape(3,-1)
            elif line[:4] == "T_02":
                translation2 = np.array(line[6:].strip().split(" ")).astype('float32').reshape(3,-1)
            elif line[:4] == "T_03":
                translation3 = np.array(line[6:].strip().split(" ")).astype('float32').reshape(3,-1)
            elif line[:9] == "S_rect_02":
                imageSize2 = np.array(line[11:].strip().split(" ")).astype('float32')
            elif line[:9] == "S_rect_03":
                imageSize3 = np.array(line[11:].strip().split(" ")).astype('float32')
            elif line[:9] == "R_rect_02":
                rectRot2 = np.array(line[11:].strip().split(" ")).astype('float32').reshape(3,-1)
            elif line[:9] == "R_rect_03":
                rectRot3 = np.array(line[11:].strip().split(" ")).astype('float32').reshape(3,-1)
            elif line[:9] == "P_rect_02":
                camMatrix2 = np.array(line[11:].strip().split(" ")).astype('float32').reshape(3,-1)
            elif line[:9] == "P_rect_03":
                camMatrix3 = np.array(line[11:].strip().split(" ")).astype('float32').reshape(3,-1)
    
    return rotation2, rotation3, translation2, translation3, imageSize2, imageSize3, \
            rectRot2, rectRot3, camMatrix2, camMatrix3, intrinsic2, intrinsic3


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

    _labels_file = str(seq_dir_ / "labels.txt")
    headers = ['frame', 'track_id', 'type', 'truncated', 'occluded', 'alpha', 'bbox_left', 'bbox_top', 'bbox_right', 'bbox_bottom', 'height', 'width', 'length', 'x', 'y', 'z', 'yaw']
    return pd.read_csv(_labels_file, sep=' ', header=None, names=headers)



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
    blockSize = 7          # odd number, usually in range 3-11
    minDisparity = -1
    maxDisparity = 6       
    numDisparities = maxDisparity-minDisparity  # max disparity minus minDisparity, must be divisible by 16
    preFilterCap = 1
    uniquenessRatio = 1

    # affect the noise
    speckleRange = 1       # multiplied by 16 implicitly, 1 or 2 usually good
    speckleWindowSize = 54 # 50-200 range
    ###################
   
    # Creating an object of StereoBM algorithm
    # Updating the parameters based on the trackbar positions
    numDisparities =  numDisparities*16
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
    leftMatcher.setP1(8*3*blockSize**2)
    leftMatcher.setP2(32*3*blockSize**2)
    
    # Calculating disparity using the stereoBM algorithm
    leftDisparity =  leftMatcher.compute(left_img, right_img).astype(np.float32)
    distL = cv2.ximgproc.getDisparityVis(leftDisparity)

    return leftMatcher, distL




def get_Q_matrix(img_size):
    cam2, cam3, k2, k3 = readAllColorMatrices()
    # Q = calculateQManually(cam2, cam3)        # We let opencv calculate the Q matrix
    
    cam2 = cam2[:,:3]
    cam3 = cam3[:,:3]
    Tmat = np.array([0.54, 0.0, 0.0])   # From the KITTI Sensor setup, in metres 
    cvQ = np.zeros((4,4))
    cv2.stereoRectify(cameraMatrix1=cam2, cameraMatrix2=cam3, distCoeffs1=0, distCoeffs2=0,
                        imageSize=img_size, R=np.identity(3), T=Tmat, 
                        R1=None, R2=None,P1=None, P2=None, Q=cvQ)
    
    return cvQ



# small main to test the module

if __name__ == "__main__":
    print("##################################")
    print("# Running the pointcloud module #")
    print("##################################")
    

    ROOT_DIR = pathlib.Path(os.getcwd())
    DATA_DIR = ROOT_DIR / "data/final_project_2023_rect"
    SEQ_01 = DATA_DIR / "seq_01"
    SEQ_02 = DATA_DIR / "seq_01"
    SEQ_03 = DATA_DIR / "seq_01"


    ## Get the images
    left_path = SEQ_01 / "image_02/data/*.png"
    right_path = SEQ_01 / "image_03/data/*.png"
    print(left_path)
    left_images = glob.glob(str(left_path))
    right_images = glob.glob(str(right_path))
    left_images.sort()
    right_images.sort()

    # Load the disparity maps
    n_frame1 = 0
    left_img1 = left_images[n_frame1]
    right_img1 = right_images[n_frame1]
    disparity_frame1 = semiGlobalMatchMap(left_img1, right_img1)


    n_frame2 = n_frame1 + 1
    left_img2 = left_images[n_frame1]
    right_img2 = right_images[n_frame1]

    disparity_frame2 = semiGlobalMatchMap(left_img2, right_img2)

    bb_boxes = get_labels_temp()
    Q = get_Q_matrix(disparity_frame1.shape[:2])


    # Finnally the pointclouds
    clusterlist1, cluster2_list, translation = ObtainListOfPontClouds(disparity_frame1
                                                            ,n_frame1, disparity_frame2, n_frame2 ,bb_boxes, Q)
    

    # Plot the pointclouds
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)
    ax.view_init(azim=0, elev=0)
    ax.scatter(clusterlist1[0][:,0], clusterlist1[0][:,1], clusterlist1[0][:,2], c=clusterlist1[0][:,3:6]/255, s=1)
    plt.show()