import cv2
import numpy as np

# Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
# https://github.com/darylclimb/cvml_project/blob/master/projections/lidar_camera_projection/utils.py

class Box3D():
    """
    Represent a 3D box corresponding to data in labels.txt
    """

    def __init__(self, label_file_line, result=False):
        data = label_file_line.split(' ') # Line holding all the data for one object from labels.txt
        self.type = data[2]                     # 'Car', 'Pedestrian', 'Cyclist'
        self.id = int(data[1])                  # unique id for each object
        data[1:] = [float(x) for x in data[3:]] # convert rest of the data to float

        self.frame = data[0]                    # frame number
        self.truncation = data[1]               
        self.occlusion = int(data[2])    # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[3]                    # object observation angle [-pi..pi]

        # extract 2D bounding box in 0-based coordinates
        self.xmin = data[4]  # left
        self.ymin = data[5]  # top
        self.xmax = data[6]  # right
        self.ymax = data[7]  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

        # extract 3D bounding box information
        self.height = data[8]   # height
        self.width = data[9]    # width
        self.length = data[10]  # length (in meters)
        self.translation = (data[11], data[12], data[13])  # (x,y,z) location in camera coordinates
        self.ry = data[14]  # rotation around y in camera coordinates [-pi;pi]
        # if result:
        #     # ONLY TO READ RESULTS
        #     self.score = data[15]  # Only for results: Score of the prediction

    def in_camera_coordinates(self):
        def roty(rotation):
            """
            Rotation about the y-axis as defined in labels.txt
            As of https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
            """
            c = np.cos(rotation)
            s = np.sin(rotation)
            r = np.array([[c, 0, s],
                          [0, 1, 0],
                          [-s, 0, c]])
            return r
        
        # Dimensions of the 3D box
        l = self.length
        w = self.width
        h = self.height

        # 3D bounding box vertices [3, 8]
        # https://towardsdatascience.com/kitti-coordinate-transformations-125094cd42fb
        x = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2] 
        y = [0, 0, 0, 0, -h, -h, -h, -h]               
        z = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2] 
        
        box_coord = np.vstack([x, y, z])

        # Rotation
        R = roty(self.ry)  # [3, 3]
        points_3d = R @ box_coord

        # Translation
        points_3d[0, :] = points_3d[0, :] + self.translation[0]
        points_3d[1, :] = points_3d[1, :] + self.translation[1]
        points_3d[2, :] = points_3d[2, :] + self.translation[2]

        return points_3d

# =========================================================
# Projections
# =========================================================
def project_to_image(points, proj_mat):
    """
    Apply the perspective projection
    Args:
        points:     3D points in camera coordinates (3, npoints)
        proj_mat:   Projection matrix (3, 4)
    """
    points = proj_mat @ points
    points /= points[2]
    
    return points[:2]


def map_box_to_image(box, proj_mat):
    """
    Projects 3D bounding box into the image plane.
    Args:
        box (Box3D): 3D bounding box object
        proj_mat: projection matrix
    """
    # Box in camera coordinates
    points_3d = box.in_camera_coordinates()

    # Project the 3D bounding box into the image plane
    points_2d = project_to_image(points_3d, proj_mat)

    return points_2d


# =========================================================
# Utils
# =========================================================
def load_label(label_filename, result=False):
    """Function reading the labels.txt file and returning a list of Object3D
    Args:
        label_filename (str): path to the labels.txt file
    Returns:
        list: a list containing all the objects in the sequence as Object3D
    """    
    lines = [line.rstrip() for line in open(label_filename)] # read all the lines in the file
    objects = [Box3D(line, result) for line in lines] # load as list of Object3D
    
    return objects


def sortLabels(labels):
    """Function grouping all the objects of the same frame together.
    Args:
        labels (list): all the labels from all the frames in the sequence
    Returns:
        sorted (list): list of lists where index = frame number in the sequence
    """    
    sorted = {}
    frameNumber = None
    for lab in labels:
        frameNumber = lab.frame
        if frameNumber not in sorted:
            sorted[frameNumber] = []
        sorted[frameNumber].append(lab)
    
    sorted = list(sorted.values())
    
    return sorted


def readProjectionMatrix(path):
    """Function to read the projection matrix in the calib_cam_to_cam.txt file.
    Args:
        path: path to the calib_cam_to_cam.txt file
    Returns:
        prect2: the projection matrix for the left rgb cameras
    """    
    
    with open(path, 'r') as f:
        fin = f.readlines()
        for line in fin:
            if line[:9] == "P_rect_02":
                pRect2 = np.array(line[11:].strip().split(" ")).astype('float32').reshape((3, 4))
                break
    pRect2 = pRect2[:, :3]  # We don't want the translation (fourth column)
    
    return pRect2


# =========================================================
# Drawing tool
# =========================================================
def draw_projected_box3d(image, pixelCoord3dBox, obj, frame, color=(0, 255, 0), thickness=1):
    """Function to project the 3D bounding box into the image plane 
       https://en.wikipedia.org/wiki/3D_projection#:~:text=provide%20additional%20realism.-,Mathematical%20formula,-%5Bedit%5D
    Args:
        image (ndarray): the image to draw on
        pixelCoord3dBox (ndarray): the 2D points of the 3D bounding box
        color (tuple, optional): RGB color value. Defaults to (255, 255, 255).
        thickness (int, optional): thickness of the box lines. Defaults to 1.
    Returns:
        image (ndarray): the image with the 3D bounding box drawn on it
    """    
    pixelCoord3dBox = pixelCoord3dBox.astype(np.int32).transpose()
    for k in range(0, 4):
        i, j = k, (k + 1) % 4          # Gets the indices of the 4 bottom points of the box
        x1, y1 = pixelCoord3dBox[i, 0], pixelCoord3dBox[i, 1]
        x2, y2 = pixelCoord3dBox[j, 0], pixelCoord3dBox[j, 1]
        # Draw the bottom square of the box
        cv2.line(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        if k == 1:
            cv2.putText(image, str(obj.id), org=(x1, y1), fontScale=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255,0,0), thickness=1, lineType=cv2.LINE_AA)

        i, j = k + 4, (k + 1) % 4 + 4  # Gets the indices of the 4 top points of the box
        x1, y1 = pixelCoord3dBox[i, 0], pixelCoord3dBox[i, 1] 
        x2, y2 = pixelCoord3dBox[j, 0], pixelCoord3dBox[j, 1]
        # Draw the top square of the box
        cv2.line(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)   
        if k == 2:
            cv2.putText(image, str(obj.type)[:3], org=(x1, y1), fontScale=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255,0,0), thickness=1, lineType=cv2.LINE_AA)

        i, j = k, k + 4   # Gets the indices of the 4 lines between the top and bottom squares
        x1, y1 = pixelCoord3dBox[i, 0], pixelCoord3dBox[i, 1]
        x2, y2 = pixelCoord3dBox[j, 0], pixelCoord3dBox[j, 1]
        # Draws the 4 lines between the top and bottom squares of the box
        cv2.line(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA) 

    return image

def drawXYZlocation(image, object, pMat):
    x,y,z = object.translation[0], object.translation[1], object.translation[2]
    # Define the 3D point
    point_3d = np.array([x, y, z])
    
    # Project the 3D point onto the 2D image using the projection matrix
    point_2d_hom = np.dot(pMat, point_3d)
    point_2d_hom /= point_2d_hom[2]
    point_2d = point_2d_hom[:2]
    cv2.circle(image, (u_obj:=int(point_2d[0]), v_obj:=int(point_2d[1])), 5, (0, 0, 255), -1) # Draw the point on the image
    cv2.putText(image, f"{str(object.type)[:3]} id: {object.id} ", org=(u_obj, v_obj), fontScale=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX,  color=(0,0,255), thickness=1, lineType=cv2.LINE_AA)

    return image

def add_frame_num(image, frame, tot_frames_):
    """
    adds frame number at bottom corner of image
    """
    cv2.putText(image, f"{str(frame)}/ {tot_frames_}", org=(50, 50), fontScale=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255,0,0), thickness=4, lineType=cv2.LINE_AA)
    return image