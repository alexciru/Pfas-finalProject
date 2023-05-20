import glob
from visualization.Box3D import *
from utils.utils import ROOT_DIR, DATA_DIR, SEQ_01, SEQ_02, SEQ_03


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
    for obj in objects[frame]:
        box3d_pixelcoord = map_box_to_image(obj, p2Matrix)
        img1 = draw_projected_box3d(img1, box3d_pixelcoord, obj, frame)
        img1 = drawXYZlocation(img1, obj, p2Matrix)

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    if results:
        cv2.imshow("results", img1)
    else:
        cv2.imshow("ground truth", img1)
    cv2.waitKey(0)


if __name__ == "__main__":
    CAM_TO_CAM = DATA_DIR / "calib_cam_to_cam.txt"
    GROUND_TRUTH = SEQ_02 / "labels.txt"
    RESULTS = ROOT_DIR / "results/seq_02_results.txt"
    SEQUENCE = ROOT_DIR / "data/video_rect/seq_02/"

    p2Matrix = readProjectionMatrix(
        CAM_TO_CAM
    )  # Load the projection matrix for the left RGB camera
    gtLabels = load_label(GROUND_TRUTH)  # Load the labels to use as ground truth
    resultLabels = load_label(RESULTS, result=True)  # Load the labels of our results
    gtLabels = sortLabels(gtLabels)  # Sort the labels by frame number
    resultLabels = sortLabels(resultLabels)  # Sort the labels by frame number
    _path = str(SEQUENCE) + "/image_02/data/*.png"
    images = glob.glob(_path)  # Read the images of the sequence
    images.sort()  # Sort the images by frame number
    # Display the ground truth and the results in two separate windows
    cv2.imshow("ground truth", cv2.imread(images[0]))
    cv2.imshow("results", cv2.imread(images[0]))
    cv2.waitKey(0)  # Wait for a key press (to be able to start recording video)
    for i in range(len(resultLabels)):
        rgb = cv2.cvtColor(cv2.imread(images[i]), cv2.COLOR_BGR2RGB)
        render_image_with_boxes(rgb, gtLabels, p2Matrix, i, time=80, results=False)
        render_image_with_boxes(rgb, resultLabels, p2Matrix, i, time=80, results=True)
    cv2.destroyAllWindows()
