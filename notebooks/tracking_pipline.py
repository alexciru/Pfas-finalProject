#!/bin/env python
# Created by Jonathan Mikler on 07/May/23

import numpy as np
from pathlib import Path
from typing import Dict
from ultralytics import YOLO

from tracking.deep_sort.deep_sort import nn_matching
from tracking.deep_sort.deep_sort.detection import Detection
from tracking.deep_sort.deep_sort.tracker import Tracker
from tracking.deep_sort.tools import generate_detections as gdet

# own
from utils.utils import (SEQ_01, SEQ_02, SEQ_03, get_root_dir, get_frames)
from utils.deepsort_utils import (LABELS_DICT, 
                                  UNKNOWN_DEFAULT,
                                  DeepSortObject,
                                  resize_masks)


def get_tracking_devices(yolo_model_:Path, deepsort_model_:Path):
    assert yolo_model_.exists(), f"YOLO model not found at {yolo_model_}"
    assert deepsort_model_.exists(), f"DeepSort model not found at {deepsort_model_}"

    print("Loading models")

    encoder = gdet.create_box_encoder(deepsort_model_, batch_size=1)
    _metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.4, None)
    tracker = Tracker(_metric, n_init=1)
    detector = YOLO(yolo_model_)

    return encoder, tracker, detector

def get_track_objects(encoder_:np.ndarray, tracker_:Tracker, detector_:YOLO, frame_l_:np.ndarray)-> Dict[int, DeepSortObject]:
    
    _detections_t = detector_(frame_l_, classes=[0, 1, 2]) # Output: List of objects (one per detection in frame)
    _detections_t = _detections_t[0]
    _masks_t = resize_masks(masks=_detections_t.masks.data.numpy(),orig_shape=_detections_t.masks.orig_shape)

    if not _detections_t: return None # TODO: Handle this in main

    _OCLUSSION_THRESHOLD = 2 # timestep duration of an object not being detected before we consider it occluded

    dsobjects_t = []
    for _det, _maks in zip(_detections_t.boxes, _masks_t):
        cls = int(_det.cls[0])
        # NOTE: I'm overriding id to match "class_id" aka 0,1,etc for label
        # another note: we can't create the DeepSortObject ahead of time like this
        # because later there is no way to match deepsort's track object to these (because for the bounding boxes are slightly diff
        # from YOLO's and other tiny details)
        # so, instead, we create them dynamically in the deepsort loop like I do below

        # QUESTION: Does this mean that the code below is unnecessary there?
        _dso = DeepSortObject(
            id=int(cls), # This changes later with deepsort
            label=LABELS_DICT.get(int(cls), UNKNOWN_DEFAULT),
            confidence=float(_det.conf[0]),
            xyxy = _det.xyxy.tolist()[0],
            mask =_maks.astype(bool)
        )
        
        dsobjects_t.append(_dso)

    # 2. Pass detections to deepsort

    # 2.1 Extract features from detections.
    all_bboxes_t = [_dso.xyxy for _dso in dsobjects_t]
    features_t = encoder_(frame_l_, all_bboxes_t)

    # 2.2 make deepsort detections from features and objects to feed the tracker
    _ds_detections_t=[Detection(_dso.tlwh, _feat, int(_dso.id), _dso.confidence, _dso.mask) for _dso, _feat in zip(dsobjects_t, features_t)]

    # 2.3 predict tracks
    tracker_.predict()
    tracker_.update(_ds_detections_t)

    # 4. Process tracks
    ds_objects = {} # ds_objects is what we should return
    
    # process/save track info like bbox, class, etc
    for track in tracker_.tracks:
        # goes over all tracked objects EVER (not just in current frame)
        # we filter for the ones detected in current with the .time_since_update check
        if not track.is_tentative():
            # track can be tentative (recently created, needs more evidence aka associations in n_init+1 frames),
            # confirmed (associated for n_init+1 or more frames), or deleted (no longer tracked)
            # a new object is classified as tentative in the first n_init frames
            # https://github.com/nwojke/deep_sort/issues/48
            print(f"Track {track.track_id} is tentative")
            print(track)

            continue

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
    print("\n--- Starting tracking pipeline ---")

    _ROOT = get_root_dir()

    encoder, tracker, detector = get_tracking_devices(
        _ROOT/'models/yolo/yolov8s-seg.pt',
        _ROOT/'models/deepsort/mars-small128.pb')

    print("Models loaded")

    # detection and tracking parameters

    frames = [(_l, _r) for _l, _r in [get_frames(frame_num_=i, seq_dir_=SEQ_01) for i in range(2)]]

    for _frame_t, (_frame_l_t, _frame_r_t) in enumerate(frames):
        # 1. det tracked objects in frame t
        # ds_objects are represents the status in frame t of the tracked objects. 
        # If an object detected in t'<t and not in t, it is occluded and will have a confidence of -1 and the .occluded property will be True
        ds_objects_t = get_track_objects(encoder_=encoder, tracker_=tracker, detector_=detector, frame_l_=_frame_l_t)
        
        if ds_objects_t is None: print(f"No objects detected in frame {_frame_t}"); continue
    
    return

if __name__ == '__main__':
    main()
