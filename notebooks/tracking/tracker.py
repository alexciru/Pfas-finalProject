from pathlib import Path

from ultralytics import YOLO
from ultralytics.yolo.utils.ops import scale_image
import pandas as pd
import numpy as np

FILE = Path(__file__).resolve()
# ROOT is the Pfas-finalProject git repo
local_parent = FILE.parents[0]
ROOT = local_parent.parents[0].parents[0]  # root directory.

# from deep_sort.application_util import preprocessing
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.tools import generate_detections

import cv2
import glob


# writes deepsort tracking id, bounding box, and class to frame
def disp_track(frame, data, color=None, label_offset=0, expected=None):
    frame = frame.copy()
    # TODO (elle): change color of bbox based on track id
    label = f"{data['track_id']}:{data['type'][:3]}"
    if expected is not None:
        label = f"{data['track_id']}/{expected['track_id']}:{data['type'][:3]}/{expected['type']}"
    # baseline is line where letters sit
    font_scale = 0.3
    font_thickness = 1
    (label_width, label_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
    )
    bbox = [
        data["bbox_left"],
        data["bbox_top"],
        data["bbox_right"],
        data["bbox_bottom"],
    ]
    top_left = tuple(
        map(
            int,
            [int(bbox[0]), int(bbox[1]) - (label_height + baseline + label_offset)],
        )
    )
    top_right = tuple(map(int, [int(bbox[0]) + label_width, int(bbox[1])]))
    org = tuple(map(int, [int(bbox[0]), int(bbox[1]) - baseline]))
    default_color = (255, 0, 0)
    # bounding box
    bbox_color = default_color if color is None else color
    cv2.rectangle(
        frame,
        (int(bbox[0]), int(bbox[1])),
        (int(bbox[2]), int(bbox[3])),
        bbox_color,
        1,
    )
    # label
    label_color = default_color if color is None else color
    cv2.rectangle(frame, top_left, top_right, label_color, -1)
    cv2.putText(
        frame,
        label,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        font_thickness,
    )
    return frame


def get_labels_df(seq_dir_):
    """
    returns the labels (ground truth data) of the given sequence as a pandas dataframe
    :param seq_dir_: sequence directory (pathlib.Path)
    :return: labels dataframe (pd.DataFrame)
    """

    _labels_file = str(ROOT / seq_dir_ / "labels.txt")
    headers = [
        "frame",
        "track_id",
        "type",
        "truncated",
        "occluded",
        "alpha",
        "bbox_left",
        "bbox_top",
        "bbox_right",
        "bbox_bottom",
        "height",
        "width",
        "length",
        "x",
        "y",
        "z",
        "yaw",
    ]
    return pd.read_csv(_labels_file, sep=" ", header=None, names=headers)


# resizes masks to original frame size
def resize_masks(masks, orig_shape):
    # rearrange mask dims from (N, H, W) to (H, W, N) for scale_image
    masks = np.moveaxis(masks, 0, -1)
    # rescale masks to original image dims
    # per https://github.com/ultralytics/ultralytics/issues/561
    masks = scale_image(masks, orig_shape)
    # rearrange masks back to (N, H, W) for visualization
    masks = np.moveaxis(masks, -1, 0)
    return masks


def execute(
    data_glob=None,
    detector_model_file=None,
    save_path=None,
    disp=True,
    expected_df=None,
    debug=False,
):
    # deepsort
    # distance used for obj similarity (helps decide if two objects are the same)
    # higher the value, easier it is to think two objects are the same
    # "Samples with larger distance [than max] are considered an invalid match."
    max_cosine_distance = 0.6
    # max_cosine_distance = 1.0
    nn_budget = None
    deepsort_model_file = local_parent / "networks/mars-small128.pb"
    detector_model_file = ROOT / detector_model_file
    if save_path:
        save_path = local_parent / f"results/{save_path}"

    # encoder is what
    encoder = generate_detections.create_box_encoder(deepsort_model_file, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget
    )
    max_age = 100
    n_init = 1
    print(
        f"""
        max_cos_dist: {max_cosine_distance},
        nn_budget: {nn_budget},
        max_age: {max_age},
        n_init: {n_init}
    """
    )
    tracker = Tracker(metric, max_age=max_age, n_init=n_init)
    detector = YOLO(detector_model_file)

    if data_glob is None:
        print("No frame path pattern provided, exiting")
        return

    frame_paths = sorted(glob.glob(data_glob))
    # leave for debugging occlusions: started from frame 25
    #     new_frame_paths = []
    # for path in frame_paths:
    #     frame_num = int(path.split("/")[-1].split(".")[0])
    #     if frame_num >= 25:
    #         new_frame_paths.append(path)
    # frame_paths = new_frame_paths
    if len(frame_paths) == 0:
        print(f"No frames found at {data_glob}, exiting")
        return

    # TODO (elle): hardcoding bc can't find programatic way to get
    # supposedly detector.names should work but it returns None
    # and I can see names in detector_pred object but can't access it via detector_pred.names
    # can try to debug with detector.__dict__ and dir(detector)
    # also, I am overriding person->Pedestrian, car->Car, and bicycle->Cyclist to match labels.txt
    val2class = {
        0: "Pedestrian",
        1: "Cyclist",
        2: "Car",
        3: "motorcycle",
        4: "airplane",
        5: "bus",
        6: "train",
        7: "truck",
        8: "boat",
        9: "traffic light",
        10: "fire hydrant",
        11: "stop sign",
        12: "parking meter",
        13: "bench",
        14: "bird",
        15: "cat",
        16: "dog",
        17: "horse",
        18: "sheep",
        19: "cow",
        20: "elephant",
        21: "bear",
        22: "zebra",
        23: "giraffe",
        24: "backpack",
        25: "umbrella",
        26: "handbag",
        27: "tie",
        28: "suitcase",
        29: "frisbee",
        30: "skis",
        31: "snowboard",
        32: "sports ball",
        33: "kite",
        34: "baseball bat",
        35: "baseball glove",
        36: "skateboard",
        37: "surfboard",
        38: "tennis racket",
        39: "bottle",
        40: "wine glass",
        41: "cup",
        42: "fork",
        43: "knife",
        44: "spoon",
        45: "bowl",
        46: "banana",
        47: "apple",
        48: "sandwich",
        49: "orange",
        50: "broccoli",
        51: "carrot",
        52: "hot dog",
        53: "pizza",
        54: "donut",
        55: "cake",
        56: "chair",
        57: "couch",
        58: "potted plant",
        59: "bed",
        60: "dining table",
        61: "toilet",
        62: "tv",
        63: "laptop",
        64: "mouse",
        65: "remote",
        66: "keyboard",
        67: "cell phone",
        68: "microwave",
        69: "oven",
        70: "toaster",
        71: "sink",
        72: "refrigerator",
        73: "book",
        74: "clock",
        75: "vase",
        76: "scissors",
        77: "teddy bear",
        78: "hair drier",
        79: "toothbrush",
    }
    all_results = []
    unknown_default = "?"
    ds_id2gt_id = {}
    id2class = {}
    for frame_idx, path in enumerate(frame_paths):
        frame_idx = int(path.split("/")[-1].split(".")[0])
        frame = cv2.imread(path)
        frame_gt = frame.copy()
        if frame is None:
            print(f"image not found at {path}")
            exit(1)

        # TODO: filter by confidence?
        # detect only person, car, and bicycle
        detector_pred = detector(frame, classes=[0, 1, 2])[0]
        bboxes = []
        yolobbox2objdata = {}
        confs = []
        objs_info = []
        if detector_pred:
            boxes = detector_pred.boxes
            masks = resize_masks(
                detector_pred.masks.data.numpy(), detector_pred.masks.orig_shape
            )
            for box, mask in zip(boxes, masks):
                conf = float(box.conf[0])
                cls_val = int(box.cls[0])
                cls = val2class[cls_val]
                # box.xyxy is tlbr (top-left, bottom-right)
                # convert from tensor to list
                box = box.xyxy.tolist()[0]
                yolobbox2objdata[tuple(box)] = (cls, conf)
                confs.append(conf)
                top_left_x, top_left_y = box[0], box[1]
                width = box[2] - box[0]
                height = box[3] - box[1]
                bbox = [top_left_x, top_left_y, width, height]
                bboxes.append(bbox)
                # TODO: get segmentation pixels
                mask = mask.astype(np.bool)
                objs_info.append((bbox, cls, conf, mask))

        # get appearance features for all objects within the bboxes
        features = encoder(frame, bboxes)
        # Detection is just a wrapper for bbox + other info we want to keep track of
        detections = []
        for obj_info, feature in zip(objs_info, features):
            bbox, cls, conf, segmentation = obj_info
            detections.append(Detection(bbox, feature, cls, conf, segmentation))
        # predict tracks
        tracker.predict()
        tracker.update(detections)
        print()
        frame_results = []
        # process/save track info like bbox, class, etc
        for track in tracker.tracks:
            # track can be tentative (recently created, needs more evidence aka associations in n_init+1 frames),
            # confirmed (associated for n_init+1 or more frames), or deleted (no longer tracked)
            # a new object is classified as tentative in the first n_init frames
            # https://github.com/nwojke/deep_sort/issues/48
            if track.is_deleted():
                del id2class[track.track_id]
                del ds_id2gt_id[track.track_id]
            if track.is_confirmed():
                # change track bbox to top left, bottom right coordinates.
                bbox = list(track.to_tlbr())
                # if occluded (aka not detected in past X frames), use previous frame's class for given id
                if track.time_since_update > 2:
                    cls = id2class[track.track_id]
                    conf = "occluded"
                else:
                    cls, conf = track.get_class(), track.get_confidence()
                    id2class[track.track_id] = cls
                mask = track.get_segmentation()
                # format matches labels.txt
                # but we set unknown_default for all values deepsort is not responsible for
                data = {
                    "frame": frame_idx,
                    "track_id": track.track_id,
                    "type": cls,
                    "truncated": unknown_default,
                    "occluded": unknown_default,
                    "alpha": unknown_default,
                    "bbox_left": int(bbox[0]),
                    "bbox_top": int(bbox[1]),
                    "bbox_right": int(bbox[2]),
                    "bbox_bottom": int(bbox[3]),
                    "height": unknown_default,
                    "width": unknown_default,
                    "length": unknown_default,
                    "x": unknown_default,
                    "y": unknown_default,
                    "z": unknown_default,
                    "yaw": unknown_default,
                    "score": conf,
                }
                all_results.append(data)
                frame_results.append(data)
                # print(f"id: {track.track_id}, frame: {frame_idx}, cls: {cls}, box: {bbox}")
                if disp and not debug:
                    frame = disp_track(frame, data)
                    frame[mask] = (0, 255, 0)

        if expected_df is not None and not debug:
            exp_frame = expected_df[expected_df["frame"] == frame_idx]
            for _, row in exp_frame.iterrows():
                frame_gt = disp_track(frame_gt, row, color=(0, 255, 0))

        ids_used = set()
        id_switches = []
        for res in frame_results:
            min_dist = 10000000
            min_exp = None
            # find row in expected_df whose bbox center is closest to the current bbox center
            bbox_center = (
                res["bbox_left"] + (res["bbox_right"] - res["bbox_left"]) / 2,
                res["bbox_top"] + (res["bbox_bottom"] - res["bbox_top"]) / 2,
            )
            # TODO (elle): make this more efficient than O(n^2)
            if expected_df is not None:
                id = res["track_id"]
                exp_frame = expected_df[expected_df["frame"] == frame_idx]
                for _, row in exp_frame.iterrows():
                    exp_bbox_center = (
                        row["bbox_left"] + (row["bbox_right"] - row["bbox_left"]) / 2,
                        row["bbox_top"] + (row["bbox_bottom"] - row["bbox_top"]) / 2,
                    )
                    err = np.linalg.norm(
                        np.array(bbox_center) - np.array(exp_bbox_center)
                    )
                    if err < min_dist:
                        min_dist = err
                        min_exp = row
                exp_id = min_exp["track_id"]
                if id not in ds_id2gt_id:
                    ds_id2gt_id[id] = exp_id
                else:
                    if ds_id2gt_id[id] != exp_id:
                        print(
                            f"WARNING ID-SWITCH?: track id {id} already mapped to {ds_id2gt_id[id]}, but now mapping to {exp_id}"
                        )
                        id_switch_str = f"{id}ds: {ds_id2gt_id[id]}gt -> {exp_id}gt"
                        id_switches.append(id_switch_str)
                        if debug:
                            frame = disp_track(frame, res)
                            new_match = min_exp
                            old_match = expected_df[expected_df["frame"] == frame_idx]
                            old_match = old_match[
                                old_match["track_id"] == ds_id2gt_id[id]
                            ]
                            old_match = old_match.to_dict("records")[0]
                            frame_gt = disp_track(
                                frame_gt, old_match, color=(0, 255, 0)
                            )
                            frame_gt = disp_track(
                                frame_gt, new_match, color=(0, 255, 0)
                            )
                        ds_id2gt_id[id] = exp_id
                if exp_id in ids_used:
                    # print(f"WARNING: track id {exp_id} already used for this frame")
                    # TODO (elle): if already mapped a DS id to this GT id, then we should check if the new DS id is closer to the GT id than the old one
                    # and override it if so
                    pass
                else:
                    ids_used.add(exp_id)
                mismatch_fmt = f"{res['type']}_ds -> {min_exp['type']}_gt "
                print(
                    f"{frame_idx}: {id} -> {exp_id}, {mismatch_fmt if res['type'] != min_exp['type'] else ''}with error {min_dist}, conf {res['score']}"
                )
        if expected_df is not None:
            exp_frame = expected_df[expected_df["frame"] == frame_idx]
            # get missed_detection (i.e. ground-truth ids we did not map any DS id to)
            gt_ids = set(exp_frame["track_id"])
            ds_ids = set(ds_id2gt_id.values())
            missed_detection_ids = list(gt_ids - ds_ids)
            missed_detections = []
            for _, row in exp_frame.iterrows():
                if row["track_id"] in missed_detection_ids:
                    missed_detections.append((row["track_id"], row["type"]))
                    if debug:
                        frame_gt = disp_track(frame_gt, row, color=(0, 0, 255))
            print(f"missed_detections {len(missed_detections)}: {missed_detections}")
            print(f"id switched {len(id_switches)}: {id_switches}")
        if disp:
            # show frame and frame_gt one on top of the other
            frame = np.concatenate((frame, frame_gt), axis=0)
            cv2.imshow("Deepsort", frame)
            # break the loop if 'q' is pressed
            if cv2.waitKey(0) & 0xFF == ord("q"):
                break
    if save_path:
        with open(save_path, "w") as f:
            for data in all_results:
                template = "{frame},{track_id},{type},{truncated},{occluded},{alpha},{bbox_left},{bbox_top},{bbox_right},{bbox_bottom},{height},{width},{length},{x},{y},{z},{yaw},{score}"
                data_fmt = template.format(**data)
                f.write(f"{data_fmt}\n")


if __name__ == "__main__":
    seq = "seq_02"
    subseq = "image_02"
    full_seq = f"{seq}/{subseq}"
    video_dir = "video_rect"
    expected_df = get_labels_df(f"data/{video_dir}/{seq}")
    # 0000000027 - 0000000055 for occlusion test
    data_glob = f"data/{video_dir}/{full_seq}/data/*.png"
    # save_path = f"track_{seq}_{subseq}.txt"
    save_path = None
    # detector_model_file = "models/yolov8n.pt"
    detector_model_file = "models/yolov8s-seg.pt"
    # detector_model_file = "/Users/ellemcfarlane/Documents/dtu/Perception_AF/Pfas/final_project/runs/detect/train2/weights/best.onnx"
    disp = True
    debug = False
    params = {
        "data_glob": data_glob,
        "save_path": save_path,
        "detector_model_file": detector_model_file,
        "disp": disp,
        "expected_df": expected_df,
        "debug": debug,
    }
    execute(**params)
