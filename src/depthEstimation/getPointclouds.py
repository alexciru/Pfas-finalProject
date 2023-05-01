import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt


def readAllColorMatrices(path):
    """Function to read all the interesting matrices in the calib_cam_to_cam.txt file
    Args:
        path: path to the calib_cam_to_cam.txt file
    Returns:
        matrices: all the relevant matrices for the colored images
    """    
    # path = "../data/final_project_2023_rect/calib_cam_to_cam.txt"
    
    with open(path, 'r') as f:
        fin = f.readlines()
        for line in fin:
            if line[:9] == "P_rect_02":
                camMatrix2 = np.array(line[11:].strip().split(" ")).astype('float32').reshape(3,-1)
            elif line[:9] == "P_rect_03":
                camMatrix3 = np.array(line[11:].strip().split(" ")).astype('float32').reshape(3,-1)
    
    return camMatrix2, camMatrix3


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





# READ IN ALL THE DATA
DATASET_PATH = "../data/final_project_2023_rect/"
DATASET_NOT_REC = "final_project_data/"

## Get the images
left_images = glob.glob(DATASET_PATH + "/seq_01/image_02/data/*.png")
right_images = glob.glob(DATASET_PATH + "/seq_01/image_03/data/*.png")
left_images.sort()
right_images.sort()

# Check that we have the images
assert(len(left_images) == len(right_images) and len(left_images) > 0)

frame = 0
left_img = cv2.imread(left_images[frame])
right_img = cv2.imread(right_images[frame])
# Blur det images to reduce noise for the stereo matching
left_img_blur = cv2.blur(cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY), (5,5))
right_img_blur = cv2.blur(cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY), (5,5))

# Create the sgbm object with known good values
numDisp, minDisp, blockS, unique, speckR, speckWindow = 6*16, 5, 7, 1, 2, 224
P1, P2 = 8*3*blockS**2, 32*2*blockS**2

sgbmObject = cv2.StereoSGBM_create(numDisparities=numDisp, minDisparity=minDisp, blockSize=blockS,
                                   uniquenessRatio=unique, speckleRange=speckR, 
                                   speckleWindowSize=speckWindow, 
                                   P1=8*3*blockS**2, P2=32*3*blockS**2)

disparityMap =  sgbmObject.compute(left_img_blur, right_img_blur).astype(np.float32)
visualMap = cv2.ximgproc.getDisparityVis(disparityMap)


# Calculate the disparity map and apply WLS filter
leftMatcher = sgbmObject
rightMatcher = cv2.ximgproc.createRightMatcher(leftMatcher)

# Calculating disparity using the stereoBM algorithm
leftDisparity =  leftMatcher.compute(left_img_blur, right_img_blur)
rightDisparity = rightMatcher.compute(right_img_blur, left_img_blur)

# Create a WLS (weighted least squares) filter (source: https://docs.opencv.org/3.4/d3/d14/tutorial_ximgproc_disparity_filtering.html) 
wlsFilter = cv2.ximgproc.createDisparityWLSFilter(leftMatcher)
wlsFilter.setLambda(8000)     # The tuning parameter, depends on the range of disparity values
wlsFilter.setSigmaColor(0.7)    # Adjusts the filter's sensitivity to edges in the image (between 0.8 and 2.0 usually good)
filteredDisparity = wlsFilter.filter(leftDisparity, left_img_blur, None, rightDisparity)

# Get the original and filtered disparity images
orgDistL = cv2.ximgproc.getDisparityVis(leftDisparity)
filteredL = cv2.ximgproc.getDisparityVis(filteredDisparity)
