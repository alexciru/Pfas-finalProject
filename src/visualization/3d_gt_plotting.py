import os
import glob
import matplotlib.pyplot as plt
import open3d
from Box3D import *

def render_image_with_boxes(img, objects, calib, frame, time=10):
    """
    Show image with 3D boxes
    """
    # projection matrix
    P_rect2cam2 = calib # left rgb camera

    img1 = np.copy(img)
    for obj in objects[frame]:
        # if obj.type == 'DontCare':
        #     continue
        box3d_pixelcoord = map_box_to_image(obj, P_rect2cam2)
        img1 = draw_projected_box3d(img1, box3d_pixelcoord)

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    cv2.imshow("img", img1)
    cv2.waitKey(time)


if __name__ == '__main__':
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    # Load calibration
    calib = readProjectionMatrices(ROOT_DIR + '/data/video_rect/calib_cam_to_cam.txt')
    # Load labels
    labels = load_label(ROOT_DIR + '/data/video_rect/seq_01/labels.txt')
    labels = load_label(ROOT_DIR + '/results.txt')
    if(len(labels)==0):
        print("No labels loaded")
    labels = sortLabels(labels)
    # Load images
    images = glob.glob(ROOT_DIR + '/data/video_rect/seq_01/image_02/data/*.png')
    images.sort()
    for i in range(len(labels)):
        rgb = cv2.cvtColor(cv2.imread(images[i]), 
                           cv2.COLOR_BGR2RGB)
        render_image_with_boxes(rgb, labels, calib, i, time=120)
    cv2.waitKey(0)
    cv2.destroyAllWindows()