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
from ResultSaving import write_results_to_file


from utils.utils import (
    ROOT_DIR,
    DATA_DIR,
    SEQ_01,
    SEQ_02,
    SEQ_03,
    get_frames,)
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


def get_track_objects(
    encoder_: np.ndarray,
    tracker_: Tracker,
    detector_: YOLO,
    frame_l_: np.ndarray,
    frame_t: int,
) -> Dict[int, DeepSortObject]:
    print("Detecting objects")
    _results_t = detector_.predict(
        frame_l_, classes=[0, 1, 2]
    )  # Output: List of objects (one per detection in frame)
    _results_t = _results_t[0]
    _masks_t = resize_masks(
        masks=_results_t.masks.data.numpy(), orig_shape=_results_t.masks.orig_shape
    )

    if not _results_t:
        return None

    # IMPORTANT: this should never be greater than MAX_AGE set for the tracker model
    _OCCLUSION_THRESHOLD = 1  # timestep duration of an object not being detected before we consider it occluded

    _detections_t = []
    for _result, _maks in zip(_results_t.boxes, _masks_t):
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
            mask=_maks.astype(bool),
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
        if frame_t != 0 and track.is_tentative():
            # track can be tentative (recently created, needs more evidence aka n_init frame associations),
            # confirmed (associated for n_init+ frames), or deleted (not seen for max_age+1 frames)
            # https://github.com/nwojke/deep_sort/issues/48
            print(
                f"Track {track.track_id}:{cls} is newly detected, not tracked until seen n_init times"
            )
            continue
        elif frame_t == 0:
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


def main():
    print("\n--- Starting pipeline ---")

    # global variables
    FRAMES = [
        (_l, _r)
        for _l, _r in [get_frames(frame_num_=i, seq_dir_=SEQ_01) for i in range(2)]
    ]

    KINEMATIC_PREDICTION = True  # True if prediction by DeepSort
    DS_PREDICTION = not KINEMATIC_PREDICTION  # True if prediction by kinematics

    # disparity map and 3d reconstruction variables
    MIN_PCD_SIZE = (
        1000  # minimum number of points in a pointcloud to be considered valid
    )
    Q = depth_est.get_Q_matrix(
        FRAMES[0][0].shape[:2], DATA_DIR / "calib_cam_to_cam.txt"
    )

    # dynamic variables
    object_tracker = ObjectTracker()

    lastFrameIds = {}  # List of objects detected in last frame

    print("Loading models")
    encoder, ds_tracker, ds_detector = get_tracking_devices(
        ROOT_DIR / "models/yolo/yolov8s-seg.pt",
        ROOT_DIR / "models/deepsort/mars-small128.pb",
    )
    print("Models loaded")

    for _frame_t, (_frame_l_t, _frame_r_t) in enumerate(FRAMES):
        print("\nAnalyzing frame", _frame_t)
        # 1. det tracked objects in frame t
        # ds_objects are represents the status in frame t of the tracked objects.
        # If an object detected in t'<t and not in t, it is occluded and will have a confidence of -1 and the .occluded property will be True
        ds_objs_t = get_track_objects(
            encoder_=encoder,
            tracker_=ds_tracker,
            detector_=ds_detector,
            frame_l_=_frame_l_t,
            frame_t=_frame_t,
        )

        if ds_objs_t is None:
            print(f"No objects detected in frame {_frame_t}")
            continue

        disparity_frame, __ = depth_est.semiGlobalMatchMap(_frame_l_t, _frame_r_t)

        _pointclouds_t = (
            dict()
        )  # key: object id, value: pointcloud of object #TODO: implement this logic, and deprecate the _objs_to_reconstruct_t list
        if _frame_t == 0:
            for _obj_id, _obj in ds_objs_t.items():
                _possible_pc = depth_reg.pointclouds_from_masks(
                    disparity_frame, _frame_l_t, [_obj.mask], Q, MIN_PCD_SIZE
                )
                if _possible_pc:
                    print(
                        f"Frame {_frame_t} | adding object {_obj} to  3D reconstruction"
                    )
                    _pointclouds_t[_obj_id] = _possible_pc[0]
        else:
            for _past_obj_id in lastFrameIds:
                # object position estimation
                if (
                    _past_obj_id in ds_objs_t.keys()
                ):  # object in last frame is in current frame
                    _obj_t = ds_objs_t.get(_past_obj_id)
                    cls = _obj_t.label
                    _possible_pc = depth_reg.pointclouds_from_masks(
                        disparity_frame, _frame_l_t, [_obj_t.mask], Q, MIN_PCD_SIZE
                    )
                    if _possible_pc:
                        print(
                            f"Frame {_frame_t} | adding object {_obj}:{cls} to  3D reconstruction"
                        )
                        _pointclouds_t[_obj_t.id] = _possible_pc[0]

                # else: # object in last frame is not in current frame (OCCLUSION)
                #     if KINEMATIC_PREDICTION:
                #         # kinematics estimation
                #         _pos_obj = object_tracker.predict_position(_past_obj_id, _frame_t)
                #     else:
                #         # deepsort estimation
                #         # get predicted bbox form deepsort
                #         raise NotImplementedError("DeepSort prediction not implemented")
                #         ...
            deleted_ids = lastFrameIds - set(ds_objs_t.keys())
            print(f"IDs no longer tracked: {deleted_ids}")
            print(f"Newly tracking IDs: {set(ds_objs_t.keys()) - lastFrameIds}")
            occluded_objs = [ob.id for ob in ds_objs_t.values() if ob.occluded]
            print(f"Occluded objects: {occluded_objs}")

        # 3. reconstruct objects in 3D
        # for pc_id, pc in _pointclouds_t.items():
        #     print(f"Object {pc_id} has {len(pc.points)} points")
        #     o3d.visualization.draw_geometries([pc], window_name=f"Frame {_frame_t}, Object {pc_id}")
        # o3d.visualization.draw_geometries(list(_pointclouds_t.values()), window_name=f"Frame {_frame_t}")

        # object tracking
        for _obj_t, _obj_pcd in _pointclouds_t.items():
            # raise NotImplementedError("3D tracking not implemented yet")

            _obj_central_position = np.mean(np.asarray(_obj_pcd.points), axis=0)
            object_tracker.update_position(_obj_t, _frame_t, _obj_central_position)

        lastFrameIds = set(ds_objs_t.keys())
    
        # save results to .txt file
        write_results_to_file(_frame_t, ds_objs_t, _pointclouds_t, object_tracker, "results.txt")


    return


if __name__ == "__main__":
    main()
