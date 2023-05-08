#!/bin/env python
# Created by Jonathan Mikler on 07/May/23

import time
from datetime import datetime
import numpy as np
from pathlib import Path
from typing import Dict, List
from ultralytics import YOLO
from tqdm import tqdm
import open3d as o3d


# own
from tracking.deep_sort.deep_sort import nn_matching
from tracking.deep_sort.deep_sort.detection import Detection
from tracking.deep_sort.deep_sort.tracker import Tracker, Track
from tracking.deep_sort.tools import generate_detections as gdet

import depth.estimation as depth_est
import depth.registration as depth_reg
import ResultSaving as results
from visualization.Box3D import project_to_image


from utils.utils import (
    ROOT_DIR,
    DATA_DIR,
    SEQ_01,
    SEQ_02,
    SEQ_03,
    get_frames,
    get_all_frames)

from utils.ObjectTracker import ObjectTracker
from utils.deepsort_utils import (
    LABELS_DICT,
    UNKNOWN_DEFAULT,
    DeepSortObject,
    resize_masks,
    )


def get_tracking_devices(yolo_model_: Path, deepsort_model_: Path):
    assert yolo_model_.exists(), f"YOLO model not found at {yolo_model_}"
    assert deepsort_model_.exists(), f"DeepSort model not found at {deepsort_model_}"

    encoder = gdet.create_box_encoder(deepsort_model_, batch_size=1)
    _metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.4, None)
    # max_age determines # consecutive frames an object is not detected before it is deleted
    # e.g. if miss an obj detection for max_age+1 times, it stops being tracked

    # TODO: determine our largest possible occlusion (in frames) and set max_age accordingly
    # alternatively, we can set max_age to a very large number and keep predicting objects that go off the screen
    tracker = Tracker(_metric, n_init=0, max_age=100)
    detector = YOLO(yolo_model_)

    return encoder, tracker, detector


def get_track_objects(encoder_:np.ndarray, tracker_:Tracker, detector_:YOLO, frame_l_:np.ndarray, frame_t:int, first_frame_:int) -> Dict[int, DeepSortObject]:
    print("Detecting objects")
    _results_t = detector_.predict(frame_l_, classes=[0, 2])  # Output: List of objects (one per detection in frame)
    _results_t = _results_t[0]
    _masks_t = resize_masks(masks=_results_t.masks.data.numpy(), orig_shape=_results_t.masks.orig_shape)

    if not _results_t:
        return None

    # IMPORTANT: this should never be greater than MAX_AGE set for the tracker model
    _OCCLUSION_THRESHOLD = 1  # timestep duration of an object not being detected before we consider it occluded

    _detections_t = []
    for _result, _mask in zip(_results_t.boxes, _masks_t):
        # NOTE: I'm overriding id to match "class_id" aka 0,1,etc for label
        # another note: we can't create the DeepSortObject ahead of time like this
        # because later there is no way to match deepsort's track object to these (because for the bounding boxes are slightly diff
        # from YOLO's and other tiny details)
        # so, instead, we create them dynamically in the deepsort loop like I do below

        # QUESTION: Does this mean that the code below is unnecessary there?
        _cls = int(_result.cls[0])
        _dso = DeepSortObject(
            id=int(_cls),  # This changes later with deepsort,
            cls=_cls,
            confidence=float(_result.conf[0]),
            xyxy=_result.xyxy.tolist()[0],
            mask=_mask.astype(bool),
        )

        _detections_t.append(_dso)

    # 2. Prepare detections to deepsort

    # 2.1 Extract features from detections.
    all_bboxes_t = [_dso.xyxy for _dso in _detections_t]
    features_t = encoder_(frame_l_, all_bboxes_t)

    # 2.2 make deepsort detections from features and objects to feed the tracker
    _ds_detections_t = [
        Detection(_dso.tlwh, _feat, int(_dso.id), _dso.confidence, _dso.mask)
        for _dso, _feat in zip(_detections_t, features_t)
    ]

    # 2.3 predict tracks
    tracker_.predict()
    tracker_.update(_ds_detections_t)

    # 4. Process tracks
    ds_objects = {}  # ds_objects is what we should return

    # process/save track info like bbox, class, etc
    for track in tracker_.tracks:
        # goes over all tracked objects EVER (not just in current frame)
        # we filter for the ones detected in current with the .time_since_update check
        # return all track (will all be tentative) for first frame,
        # otherwsie skip tracks that are not confirmed
        cls = LABELS_DICT.get(track.get_class(), UNKNOWN_DEFAULT)
        if frame_t != first_frame_ and track.is_tentative():
            # track can be tentative (recently created, needs more evidence aka n_init frame associations),
            # confirmed (associated for n_init+ frames), or deleted (not seen for max_age+1 frames)
            # https://github.com/nwojke/deep_sort/issues/48
            print(
                f"Track {track.track_id}:{cls} is newly detected, not tracked until seen n_init times"
            )
            continue
        elif frame_t == first_frame_:
            print(f"Detected objects in frame 0: {track.track_id}:{cls}")

        # if occluded (aka not detected in past X frames), use previous frame's class for given id
        if track.time_since_update >= _OCCLUSION_THRESHOLD:
            conf = -1  # -1 means occluded
        else:
            conf = track.get_confidence()

        ds_objects[track.track_id] = DeepSortObject(
            id=track.track_id,
            cls=int(track.get_class()),
            confidence=conf,
            xyxy=list(track.to_tlbr()),
            mask=track.get_segmentation().astype(bool),
        )

    return ds_objects


def main(seq_: Path, begin_frame_: int = 0, end_frame_: int = 144):
    print("\n--- Starting pipeline ---")

    RESULTS_FILENAME = ROOT_DIR / f"results/{seq_.name}_results.txt"
    LOG_FILENAME = ROOT_DIR / f"logs/{seq_.name}_log.txt"
    results.reset_file(RESULTS_FILENAME)
    results.reset_file(LOG_FILENAME)

    results.log_info(LOG_FILENAME, f"time: {datetime.now().strftime('%H:%M:%S')}")

    # global variables
    # FRAMES = [(_l, _r) for _l, _r in [get_frames(frame_num_=i, seq_dir_=seq_) for i in range(_first_frame,_last_frame+1)]]
    FRAMES = get_all_frames(seq_dir_=seq_)

    # disparity map and 3d reconstruction variables
    MIN_PCD_SIZE = (1000)  # minimum number of points in a pointcloud to be considered valid

    Q = depth_est.get_Q_matrix(
        FRAMES[0][0].shape[:2], DATA_DIR / "calib_cam_to_cam.txt"
    )
    CAM_MAT_L, CAM_MAT_R = depth_est.get_cam_matrices(DATA_DIR / "calib_cam_to_cam.txt")[9:11]

    # dynamic variables
    object_tracker = ObjectTracker()

    lastFrameIds = set()  # List of objects detected in last frame

    print("Loading models")
    encoder, ds_tracker, ds_detector = get_tracking_devices(
        ROOT_DIR / "models/yolo/yolov8s-seg.pt",
        ROOT_DIR / "models/deepsort/mars-small128.pb",
    )
    print("Models loaded")

    for _frame_t, (_frame_l_t, _frame_r_t) in enumerate(tqdm(FRAMES), begin_frame_):
        if _frame_t > end_frame_ or _frame_t < begin_frame_:
            continue
        print("\n\nAnalyzing frame", _frame_t)

        results.log_info(LOG_FILENAME, f"Analyzing frame {_frame_t} | {datetime.now().strftime('%H:%M:%S')}")
        # 1. det tracked objects in frame t
        # ds_objects are represents the status in frame t of the tracked objects.
        # If an object detected in t'<t and not in t, it is occluded and will have a confidence of -1 and the .occluded property will be True
        ds_objs_t = get_track_objects(encoder_=encoder, tracker_=ds_tracker, detector_=ds_detector, frame_l_=_frame_l_t, frame_t=_frame_t, first_frame_=begin_frame_)

        if ds_objs_t is None:
            print(f"No objects detected in frame {_frame_t}")
            continue

        results.log_info(LOG_FILENAME, f"Frame {_frame_t} | Objects detected in frame: {ds_objs_t.keys()}")

        disparity_frame, __ = depth_est.semiGlobalMatchMap(_frame_l_t, _frame_r_t)

        _pointclouds_t = dict()  # key: object id, value: pointcloud of object #TODO: implement this logic, and deprecate the _objs_to_reconstruct_t list
        if _frame_t == begin_frame_:
            for _obj_id, _obj in ds_objs_t.items():
                _possible_pc = depth_reg.pointclouds_from_masks(disparity_frame, _frame_l_t, [_obj.mask], Q, MIN_PCD_SIZE)
                if _possible_pc:
                    print(f"Frame {_frame_t} | adding object {_obj} to  3D reconstruction")
                    _pointclouds_t[_obj_id] = _possible_pc[0]

        else:
            for _past_obj_id in lastFrameIds:
                # object position estimation
                if (_past_obj_id in ds_objs_t.keys()):  # object in last frame is in current frame
                    _ds_obj_t = ds_objs_t.get(_past_obj_id)
                    _possible_pc = depth_reg.pointclouds_from_masks(disparity_frame, _frame_l_t, [_ds_obj_t.mask], Q, MIN_PCD_SIZE)
                    if _possible_pc:
                        print(f"Frame {_frame_t} | adding object {_ds_obj_t} to  3D reconstruction")
                        _pointclouds_t[_ds_obj_t.id] = _possible_pc[0]
                    else:
                        results.log_info(LOG_FILENAME, f"Frame {_frame_t} | object {_ds_obj_t.id} cloud too small. not added to 3D reconstruction")

                else: # object in last frame is not in current frame (OCCLUSION)
                    # kinematics estimation
                    _pos_obj_t = object_tracker.predict_position(_past_obj_id, _frame_t)
                    if _pos_obj_t is None:
                        results.log_info(LOG_FILENAME, f"Frame {_frame_t} | object {_past_obj_id} not found in frame {_frame_t} and not predicted")
                        continue
                    _p_image = project_to_image(_pos_obj_t, CAM_MAT_L[:,:3]) # we only care for the rotation.
                    # if _p_image in closer to the frame the do not add it to the reconstruction
                    FRAME_FENCE = 10
                    if _p_image[0] < FRAME_FENCE or _p_image[0] > _frame_l_t.shape[1]-FRAME_FENCE or _p_image[1] < FRAME_FENCE or _p_image[1] > _frame_l_t.shape[0]- FRAME_FENCE:
                        print(f"Frame {_frame_t} | object {_past_obj_id} is out of the frame")
                        results.log_info(LOG_FILENAME, f"Frame {_frame_t} | object {_past_obj_id} is out of the frame")
                        continue
                    object_tracker.update_position(time_=_frame_t, obj_key_=_past_obj_id, position_=_pos_obj_t)
            
        # QUESTION: Should this update of last objects happen also starting from frame 0?
        deleted_ids = lastFrameIds - set(ds_objs_t.keys())
        results.log_info(LOG_FILENAME, f"Frame {_frame_t} | IDs no longer tracked: {deleted_ids}")
        results.log_info(LOG_FILENAME, f"Frame {_frame_t} | Newly tracking IDs: {set(ds_objs_t.keys()) - lastFrameIds}")
        occluded_objs = [ob.id for ob in ds_objs_t.values() if ob.occluded]
        results.log_info(LOG_FILENAME,f"Frame {_frame_t} | Occluded objects: {occluded_objs}")

        # 3. reconstruct objects in 3D
        # for pc_id, pc in _pointclouds_t.items():
        #     print(f"Object {pc_id} has {len(pc.points)} points")
        #     o3d.visualization.draw_geometries([pc], window_name=f"Frame {_frame_t}, Object {pc_id}")
        # o3d.visualization.draw_geometries(list(_pointclouds_t.values()), window_name=f"Frame {_frame_t}")

        # track objects from pointclouds
        for _ds_obj_t, _obj_pcd in _pointclouds_t.items():
            # raise NotImplementedError("3D tracking not implemented yet")
            _obj_central_position = np.mean(np.asarray(_obj_pcd.points), axis=0)
            object_tracker.update_position(time_=_frame_t, obj_key_=_ds_obj_t, position_=_obj_central_position)

        results.log_info(LOG_FILENAME, f"Frame {_frame_t} | objects tracked in frame {object_tracker.objects_in_time[_frame_t].keys()}")
        results.log_info(LOG_FILENAME, f"Frame {_frame_t} | objects detected in last frame: {lastFrameIds}")
        lastFrameIds = set(ds_objs_t.keys())
    
        # save results for time t to results.txt file
        results.new_save_timeframe_results(_frame_t, object_tracker, ds_objs_t, _pointclouds_t, RESULTS_FILENAME)


    return


if __name__ == "__main__":
    main(SEQ_01, begin_frame_=0, end_frame_=144)
