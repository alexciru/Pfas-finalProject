#!/bin/env python
# Created by Jonathan Mikler on 07/May/23

import numpy as np
from pathlib import Path
from typing import Dict, List
from ultralytics import YOLO

import open3d as o3d


# own
from tracking.deep_sort.deep_sort import nn_matching
from tracking.deep_sort.deep_sort.detection import Detection
from tracking.deep_sort.deep_sort.tracker import Tracker, Track
from tracking.deep_sort.tools import generate_detections as gdet

import depth.estimation as depth_est
import depth.registration as depth_reg



from utils.utils import (ROOT_DIR, DATA_DIR, SEQ_01, SEQ_02, SEQ_03, get_frames, ObjectTracker)
from utils.deepsort_utils import (LABELS_DICT, 
                                  UNKNOWN_DEFAULT,
                                  DeepSortObject,
                                  resize_masks)


def get_tracking_devices(yolo_model_:Path, deepsort_model_:Path):
    assert yolo_model_.exists(), f"YOLO model not found at {yolo_model_}"
    assert deepsort_model_.exists(), f"DeepSort model not found at {deepsort_model_}"

    encoder = gdet.create_box_encoder(deepsort_model_, batch_size=1)
    _metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.4, None)
    tracker = Tracker(_metric, n_init=1)
    detector = YOLO(yolo_model_)

    return encoder, tracker, detector

def get_track_objects(encoder_:np.ndarray, tracker_:Tracker, detector_:YOLO, frame_l_:np.ndarray)-> Dict[int, DeepSortObject]:
    
    print("Detecting objects")
    _results_t = detector_.predict(frame_l_, classes=[0, 1, 2]) # Output: List of objects (one per detection in frame)
    _results_t = _results_t[0]
    _masks_t = resize_masks(masks=_results_t.masks.data.numpy(),orig_shape=_results_t.masks.orig_shape)

    if not _results_t: return None

    _OCLUSSION_THRESHOLD = 2 # timestep duration of an object not being detected before we consider it occluded

    _detections_t = []
    for _result, _maks in zip(_results_t.boxes, _masks_t):
        cls = int(_result.cls[0])
        # NOTE: I'm overriding id to match "class_id" aka 0,1,etc for label
        # another note: we can't create the DeepSortObject ahead of time like this
        # because later there is no way to match deepsort's track object to these (because for the bounding boxes are slightly diff
        # from YOLO's and other tiny details)
        # so, instead, we create them dynamically in the deepsort loop like I do below

        # QUESTION: Does this mean that the code below is unnecessary there?
        _dso = DeepSortObject(
            id=int(cls), # This changes later with deepsort
            label=LABELS_DICT.get(int(cls), UNKNOWN_DEFAULT),
            confidence=float(_result.conf[0]),
            xyxy = _result.xyxy.tolist()[0],
            mask =_maks.astype(bool)
        )

        _detections_t.append(_dso)

    # 2. Prepare detections to deepsort

    # 2.1 Extract features from detections.
    all_bboxes_t = [_dso.xyxy for _dso in _detections_t]
    features_t = encoder_(frame_l_, all_bboxes_t)

    # 2.2 make deepsort detections from features and objects to feed the tracker
    _ds_detections_t=[Detection(_dso.tlwh, _feat, int(_dso.id), _dso.confidence, _dso.mask) for _dso, _feat in zip(_detections_t, features_t)]

    # 2.3 predict tracks
    tracker_.predict()
    tracker_.update(_ds_detections_t)

    # 4. Process tracks
    ds_objects = {} # ds_objects is what we should return
    
    # process/save track info like bbox, class, etc
    for track in tracker_.tracks:
        # goes over all tracked objects EVER (not just in current frame)
        # we filter for the ones detected in current with the .time_since_update check
        if not track.is_confirmed(): #BUG
            # track can be tentative (recently created, needs more evidence aka associations in n_init+1 frames),
            # confirmed (associated for n_init+1 or more frames), or deleted (no longer tracked)
            # a new object is classified as tentative in the first n_init frames
            # https://github.com/nwojke/deep_sort/issues/48
            ...
            # continue

        # if occluded (aka not detected in past X frames), use previous frame's class for given id
        if track.time_since_update >= _OCLUSSION_THRESHOLD:
            # TODO: we could use this to fill in the "occluded" state instead
            conf = -1 # -1 means occluded
        else:
            conf = track.get_confidence()

        ds_objects[track.track_id] = DeepSortObject(
            id=track.track_id,
            label=LABELS_DICT.get(int(track.get_class()), UNKNOWN_DEFAULT),
            confidence=conf,
            xyxy=list(track.to_tlbr()),
            mask=track.get_segmentation().astype(bool)
        )

    return ds_objects

def main():
    print("\n--- Starting pipeline ---")

    # global variables
    FRAMES = [(_l, _r) for _l, _r in [get_frames(frame_num_=i, seq_dir_=SEQ_01) for i in range(2)]]

    KINEMATIC_PREDICTION = True # True if prediction by DeepSort
    DS_PREDICTION = not KINEMATIC_PREDICTION # True if prediction by kinematics

    # disparity map and 3d reconstruction variables
    MIN_PCD_SIZE = 1000 # minimum number of points in a pointcloud to be considered valid
    Q = depth_est.get_Q_matrix(FRAMES[0][0].shape[:2], DATA_DIR / "calib_cam_to_cam.txt")

    # dynamic variables
    object_tracker = ObjectTracker()

    lastFrameIds = {} # List of objects detected in last frame

    print("Loading models")
    encoder, ds_tracker, ds_detector = get_tracking_devices(
        ROOT_DIR/'models/yolo/yolov8s-seg.pt',
        ROOT_DIR/'models/deepsort/mars-small128.pb')
    print("Models loaded")

    for _frame_t, (_frame_l_t, _frame_r_t) in enumerate(FRAMES):
        print("\nAnalyzing frame", _frame_t)
        # 1. det tracked objects in frame t
        # ds_objects are represents the status in frame t of the tracked objects. 
        # If an object detected in t'<t and not in t, it is occluded and will have a confidence of -1 and the .occluded property will be True
        ds_objs_t = get_track_objects(encoder_=encoder, tracker_=ds_tracker, detector_=ds_detector, frame_l_=_frame_l_t)
        
        if ds_objs_t is None: print(f"No objects detected in frame {_frame_t}"); continue

        disparity_frame, __ = depth_est.semiGlobalMatchMap(_frame_l_t, _frame_r_t)
    
        _objs_to_reconstruct_t = []
        _pointclouds_t = dict() # key: object id, value: pointcloud of object #TODO: implement this logic, and deprecate the _objs_to_reconstruct_t list
        if _frame_t == 0:
            # add all objects to reconstruction
            _objs_to_reconstruct_t = list(ds_objs_t.values())
        else:
            for _past_obj_id in lastFrameIds:
                # object position estimation
                if _past_obj_id in ds_objs_t.keys(): # object in last frame is in current frame
                    print(f'Adding object {_past_obj_id} to reconstruction')
                    _objs_to_reconstruct_t.append(ds_objs_t.get(_past_obj_id))
                    # TODO: instead of adding to reconstruction, create the pc here @alex
                
                # else: # object in last frame is not in current frame (OCCLUSION)
                #     if KINEMATIC_PREDICTION:
                #         # kinematics estimation
                #         _pos_obj = object_tracker.predict_position(_past_obj_id, _frame_t)
                #     else:
                #         # deepsort estimation
                #         # get predicted bbox form deepsort
                #         raise NotImplementedError("DeepSort prediction not implemented")
                #         ...
        
        # 3. reconstruct objects in 3D
        # TODO: Creating the pcs here as a list looses track of which ps belongs to which ds_object. pc creating needs to be done at the ds_object level
        _obj_masks = [_obj.mask for _obj in _objs_to_reconstruct_t]
        _obj_pointclouds_t = depth_reg.pointclouds_from_masks(disparity_frame, _frame_l_t, _obj_masks, Q, MIN_PCD_SIZE)

        o3d.visualization.draw_geometries(_obj_pointclouds_t, window_name=f"Frame {_frame_t}")

        # object tracking
        for _obj_pcd in _obj_pointclouds_t:
            # raise NotImplementedError("3D tracking not implemented yet")

            _obj_central_position = np.mean(np.asarray(_obj_pcd.points), axis=0)

            object_tracker.register_position(_past_obj_id, _frame_t, _pos_obj)

        lastFrameIds = set(ds_objs_t.keys())
    return

if __name__ == '__main__':
    main()
