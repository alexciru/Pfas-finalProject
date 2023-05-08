import numpy as np
from utils.utils import (
    ROOT_DIR,
    DATA_DIR,
    SEQ_01,
    SEQ_02,
    SEQ_03,
    get_frames,
    get_labels_df,
    get_df,
)

from utils.deepsort_utils import (
    LABELS_DICT,
    UNKNOWN_DEFAULT,
)


def find_match(obj, gt_track_objs):
    """
    finds the closest match to obj in gt_track_objs
    :param obj: object to match (pd.Series)
    :param gt_track_objs: ground truth objects (pd.DataFrame)
    :return: closest match (pd.Series), error (float)
    """
    min_err = 10000000
    closest_match = None
    bbox_target = (
        obj["bbox_left"] + (obj["bbox_right"] - obj["bbox_left"]) / 2,
        obj["bbox_top"] + (obj["bbox_bottom"] - obj["bbox_top"]) / 2,
    )
    # find ground-truth obj with closest bbox center to obj's
    for _, gt_obj in gt_track_objs.iterrows():
        gt_bbox_center = (
            gt_obj["bbox_left"] + (gt_obj["bbox_right"] - gt_obj["bbox_left"]) / 2,
            gt_obj["bbox_top"] + (gt_obj["bbox_bottom"] - gt_obj["bbox_top"]) / 2,
        )
        # euclidian distance between bbox centers
        err = np.linalg.norm(np.array(bbox_target) - np.array(gt_bbox_center))
        if err < min_err:
            min_err = err
            closest_match = gt_obj
    return closest_match, min_err


if __name__ == "__main__":
    results_file = ROOT_DIR / "results/seq_01_results.txt"
    video_dir = "video_rect"
    expected_df = get_labels_df(SEQ_01)
    actual_df = get_df(results_file)
    ds_id2gt_id = {}
    # frame -> id_switch_str
    id_switches = {}
    n_correct_ids = 0
    # frame-> id_mismatches_str
    misclass = {}
    n_correct_class = 0
    # LABELS_DICT.get(self.cls, UNKNOWN_DEFAULT)
    # ERR_THRESH = 30
    # for each gt_track_objs, track_objs in results df and ground truth df
    for frame, ((_, gt_track_objs), (_, track_objs)) in enumerate(
        zip(expected_df.groupby("frame"), actual_df.groupby("frame"))
    ):
        # for each obj in frame_res, find closest bounding box match in ground df
        print(f"frame: {frame}")
        for _, obj in track_objs.iterrows():
            ds_id = int(obj["track_id"])
            ds_cls = LABELS_DICT.get(obj["type"], UNKNOWN_DEFAULT)
            gt_obj_match, err = find_match(obj, gt_track_objs)
            gt_id = gt_obj_match["track_id"]
            gt_cls = gt_obj_match["type"]
            print(f"{ds_id}->{gt_id}:{ds_cls}->{gt_cls}, err: {err}pixels")
        print()
        if frame == 10:
            exit(1)
        # if err <= ERR_THRESH:
        # if obj id not yet in ds_id2gt_id, add it
        # if it is, check if gt_match is the same as the one in ds_id2gt_id
        # and if not add to
        # id_switch_str = f"{id}ds: {ds_id2gt_id[id]}gt -> {exp_id}gt"
        # id_switches[frame]=(id_switch_str)
        # also check if mismatch bet gt_match.class and obj.class
        # USE LABELS_DICT for our res
        # if don't match, add to mismatches
    # for each obj in frame_res, find closest bounding box match in ground df
    # gt_match, err = find_match(obj, gt_track_objs)

    # if err <= ERR_THRESH:
    # if obj id not yet in ds_id2gt_id, add it
    # if it is, check if gt_match is the same as the one in ds_id2gt_id
    # and if not add to
    # id_switch_str = f"{id}ds: {ds_id2gt_id[id]}gt -> {exp_id}gt"
    # id_switches.append(id_switch_str)
    # also check if mismatch bet gt_match.class and obj.class
    # USE LABELS_DICT for our res
    # if don't match, add to mismatches

    # after id loop, check for missed detections:
    # i.e. whatever.

    # print len(id_switches)/len(id_switches)+n_correct_ids for % correctly tracked
    # OF the ones we tracked
    # print len(misclass)/len(misclass)+n_correct_class for % correctly classified
