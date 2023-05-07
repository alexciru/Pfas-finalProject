import glob
from Box3D import *

def render_image_with_boxes(img, objects, p2Matrix, frame, time=10, results=False):
    """Function to render the image with the 3D bounding boxes.
    Args:
        img (ndarray): the rgb image to render the boxes on
        objects (list): list of all the objects in the sequence, of type Object3D
        p2Matrix (ndarray): the projection matrix for the left rgb camera
        frame (int): the frame number in the sequence 
        time (int, optional): how long to display. Defaults to 10.
    """    
    img1 = np.copy(img)
    res = 100 if results else 0
    for obj in objects[frame]:
        box3d_pixelcoord = map_box_to_image(obj, p2Matrix)
        img1 = draw_projected_box3d(img1, box3d_pixelcoord, obj.id, results=res)
        img1 = drawXYZlocation(img1, obj, p2Matrix, results=res)

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    cv2.imshow("img", img1)
    cv2.waitKey(time)

if __name__ == '__main__':
    # Load the projection matrix for the left RGB camera
    p2Matrix = readProjectionMatrix('../data/final_project_2023_rect/calib_cam_to_cam.txt')
    # Load the labels to use as ground truth
    labels = load_label('../data/final_project_2023_rect/seq_01/labels.txt'); results = False
    labels = load_label('results2.txt', result=True); results = True
    labels = sortLabels(labels)
    # Read the images of the sequence
    images = glob.glob('../data/final_project_2023_rect/seq_01/image_02/data/*.png')
    images.sort()
    cv2.imshow("img", cv2.imread(images[0]))
    cv2.waitKey(0)
    for i in range(len(labels)):
        rgb = cv2.cvtColor(cv2.imread(images[i]), cv2.COLOR_BGR2RGB)
        render_image_with_boxes(rgb, labels, p2Matrix, i, time=100, results=results)
    cv2.destroyAllWindows()