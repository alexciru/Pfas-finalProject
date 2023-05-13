import numpy as np
from utils.utils import (
    ROOT_DIR,
    DATA_DIR,
    SEQ_01,
    SEQ_02,
    SEQ_03,
    get_labels_df,
    get_our_df,
)

from utils.deepsort_utils import LABELS_DICT, UNKNOWN_DEFAULT, disp_track

# import matplotlib.pyplot as plt
import plotly.graph_objects as go
import cv2

import dash
import dash_core_components as dcc
import dash_html_components as html

import requests


def find_match(obj, other_objs):
    """
    finds the closest match to obj in other_objs
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
    for _, gt_obj in other_objs.iterrows():
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


def threeD_err(obj, gt_obj):
    obj_center = np.array([obj["x"], obj["y"], obj["z"]])
    gt_obj_center = np.array([gt_obj["x"], gt_obj["y"], gt_obj["z"]])
    return np.linalg.norm(obj_center - gt_obj_center)


def update_data(
    fig,
    ground_truth2deep_sort,
    marker_data_pred,
    marker_data_expected,
    expected_x,
    expected_y,
    predicted_x,
    predicted_y,
):
    for frame_idx, frame_path in enumerate(frame_paths):
        if frame_idx < len(ground_truth2deep_sort):
            path = str(frame_path)
            frame = cv2.imread(path)
            frame_gt = frame.copy()
            frame_ds = frame.copy()
            if frame is None:
                print(f"image not found at {path}")
                exit(1)
            gt_maps = ground_truth2deep_sort[frame_idx]
            if GT_ID_OF_INTEREST in gt_maps:
                data = gt_maps[GT_ID_OF_INTEREST]
                maps_of_interest.append(data)
                gt_obj = data["gt_obj"]
                expected_x = np.append(expected_x, float(gt_obj["x"]))
                expected_y = np.append(expected_y, float(gt_obj["y"]))
                ds_obj = data["ds_obj"]
                err = data["err"]
                ds_cls = ds_obj["type"] if ds_obj is not None else "None"
                gt_cls = gt_obj["type"]
                ds_id = ds_obj["track_id"] if ds_obj is not None else "None"
                gt_id = gt_obj["track_id"]
                if ds_obj is not None:
                    frame_ds = disp_track(frame, ds_obj)
                    predicted_x = np.append(predicted_x, float(ds_obj["x"]))
                    predicted_y = np.append(predicted_y, float(ds_obj["y"]))
                else:
                    predicted_x = np.append(predicted_x, np.nan)
                    predicted_y = np.append(predicted_y, np.nan)
                frame_gt = disp_track(frame, gt_obj)
                print(
                    f"frame: {frame_idx}, {gt_id}->{ds_id}, {gt_cls}->{ds_cls}, err: {err}"
                )
            else:
                expected_x = np.append(expected_x, np.nan)
                predicted_x = np.append(predicted_x, np.nan)
                expected_y = np.append(expected_y, np.nan)
                predicted_y = np.append(predicted_y, np.nan)

            # update traces in plotly
            marker_data_pred.x = time
            marker_data_pred.y = predicted_x
            marker_data_pred.z = predicted_y
            marker_data_expected.x = time
            marker_data_expected.y = expected_x
            marker_data_expected.z = expected_y
            fig.update_traces(marker_data_pred, selector=dict(name="predicted"))
            fig.update_traces(marker_data_expected, selector=dict(name="expected"))

            #### SHOW RESULTS
            frame = np.concatenate((frame_ds, frame_gt), axis=0)
            cv2.imshow("Deepsort", frame)
            # break the loop if 'q' is pressed
            if cv2.waitKey(0) & 0xFF == ord("q"):
                break  ############ PLOT RESULTS
    # start the dash server


if __name__ == "__main__":
    results_file = ROOT_DIR / "results/seq_02_results.txt"
    video_dir = "video_rect"
    expected_df = get_labels_df(SEQ_02)
    data_glob = SEQ_02 / "image_02/data"
    frame_paths = sorted(data_glob.glob("*.png"))
    actual_df = get_our_df(results_file)
    gt2ds_ids = {}
    # frame -> id_switch_str
    id_switches = {}
    n_id_switches = 0
    n_misclass = 0
    n_same_obj = 0
    # frame-> id_mismatches_str
    misclass = {}
    n_correct_class = 0
    # LABELS_DICT.get(self.cls, UNKNOWN_DEFAULT)
    ERR_THRESH = 100
    GT_ID_OF_INTEREST = 7
    predicted = {}
    expected = {}
    errs = {}
    # maps frame->gt_id->{"gt_obj": obj, "ds_obj": ds_obj, "ds_id": ds_id, "err": err}
    ground_truth2deep_sort = {}
    matches = {}
    for frame, ((_, gt_track_objs), (_, track_objs)) in enumerate(
        zip(expected_df.groupby("frame"), actual_df.groupby("frame"))
    ):
        # if frame >= 5:
        #     break
        if frame not in ground_truth2deep_sort:
            ground_truth2deep_sort[frame] = {}
        # for each obj in frame_res, find closest bounding box match in ground df
        for _, gt_obj in gt_track_objs.iterrows():
            gt_id = int(gt_obj["track_id"])
            gt_cls = gt_obj["type"]
            ds_obj, err = find_match(gt_obj, track_objs)
            ds_id = ds_obj["track_id"]
            ds_cls = ds_obj["type"]
            probably_same_obj = err <= ERR_THRESH
            # condition for basically getting the same object
            if probably_same_obj:
                n_same_obj += 1
                err = threeD_err(gt_obj, ds_obj)
                if gt_id not in ground_truth2deep_sort[frame]:
                    ground_truth2deep_sort[frame][gt_id] = {
                        "gt_obj": gt_obj,
                        "ds_obj": ds_obj,
                        "ds_id": ds_id,
                        "err": err,
                    }
                # override map if err is smaller
                elif err < ground_truth2deep_sort[frame][gt_id]["err"]:
                    ground_truth2deep_sort[frame][gt_id] = {
                        "gt_obj": gt_obj,
                        "ds_obj": ds_obj,
                        "ds_id": ds_id,
                        "err": err,
                    }
                if gt_id not in gt2ds_ids:
                    gt2ds_ids[gt_id] = {ds_id}
                # if we map gt to another ds_id then we have an id switch
                elif ds_id not in gt2ds_ids[gt_id]:
                    gt2ds_ids[gt_id].add(ds_id)
                    id_switch_str = f"new ds_id mapped to same gt_id: {gt_id}->{ds_id}\nall ds_ids: {gt2ds_ids[gt_id]}"
                    id_switches[frame] = id_switch_str
                    n_id_switches += 1
                # if we think we've mapped IDs correctly, then check if we've classified correctly too
                if ds_cls != gt_cls:
                    misclass_str = f"{ds_id}ds: {ds_cls} -> {gt_cls}gt"
                    misclass[frame] = misclass_str
                    n_misclass += 1
            else:
                # no match found under err threshold
                ground_truth2deep_sort[frame][gt_id] = {
                    "gt_obj": gt_obj,
                    "ds_obj": None,
                    "ds_id": np.nan,
                    "err": np.nan,
                }
    # percentage of id switches out of # obj we think match to gt
    percent_switch = 100 * n_id_switches / n_same_obj
    print(f"percent id switches: {percent_switch}%")
    # percentage of misclassifications
    percent_miscl = 100 * n_misclass / n_same_obj
    print(f"percent misclassifications: {percent_miscl}%")

    # print(ground_truth2deep_sort)
    # TODO: refactor to hold x,y in same array

    time = [i for i in range(0, len(ground_truth2deep_sort))]
    print(len(time))
    maps_of_interest = []

    # SANITY CHECK
    # create the initial plot
    # expected_x = np.array([])
    # predicted_x = np.array([])
    # expected_y = np.array([])
    # predicted_y = np.array([])

    # marker_data_pred = go.Scatter3d(
    #     x=time,
    #     y=predicted_x,
    #     z=predicted_y,
    #     marker=go.scatter3d.Marker(size=2),
    #     mode="markers",
    #     marker_color="red",
    # )
    # marker_data_expected = go.Scatter3d(
    #     x=time,
    #     y=expected_x,
    #     z=expected_y,
    #     marker=go.scatter3d.Marker(size=2),
    #     mode="markers",
    #     marker_color="blue",
    # )
    # marker_data_pred.name = "predicted"
    # marker_data_expected.name = "expected"
    # fig = go.Figure(data=[marker_data_pred, marker_data_expected])
    # fig.update_layout(scene=dict(xaxis_title="time", yaxis_title="x", zaxis_title="y"))

    # # update the data and start the server
    print("starting server")

    expected_x = np.array([])
    predicted_x = np.array([])
    expected_y = np.array([])
    predicted_y = np.array([])

    # marker_data_pred = go.Scatter3d(
    #     x=time,
    #     y=predicted_x,
    #     z=predicted_y,
    #     marker=go.scatter3d.Marker(size=2),
    #     mode="markers",
    #     marker_color="red",
    # )
    # marker_data_expected = go.Scatter3d(
    #     x=time,
    #     y=expected_x,
    #     z=expected_y,
    #     marker=go.scatter3d.Marker(size=2),
    #     mode="markers",
    #     marker_color="blue",
    # )
    # # set trace names
    # marker_data_pred.name = "predicted"
    # marker_data_expected.name = "expected"
    # # fig = go.Figure(data=marker_data_expected)
    # fig = go.Figure(data=[marker_data_pred, marker_data_expected])
    # fig.update_layout(scene=dict(xaxis_title="time", yaxis_title="x", zaxis_title="y"))
    # app = dash.Dash()
    # app.layout = html.Div([dcc.Graph(figure=fig)])
    # app.run_server(debug=True, use_reloader=False)
    url = "http://localhost:8050/live-update-graph-scatter"
    for frame_idx, frame_path in enumerate(frame_paths):
        if frame_idx < len(ground_truth2deep_sort):
            path = str(frame_path)
            frame = cv2.imread(path)
            frame_gt = frame.copy()
            frame_ds = frame.copy()
            if frame is None:
                print(f"image not found at {path}")
                exit(1)
            gt_maps = ground_truth2deep_sort[frame_idx]
            if GT_ID_OF_INTEREST in gt_maps:
                data = gt_maps[GT_ID_OF_INTEREST]
                maps_of_interest.append(data)
                gt_obj = data["gt_obj"]
                expected_x = np.append(expected_x, float(gt_obj["x"]))
                expected_y = np.append(expected_y, float(gt_obj["y"]))
                ds_obj = data["ds_obj"]
                err = data["err"]
                ds_cls = ds_obj["type"] if ds_obj is not None else "None"
                gt_cls = gt_obj["type"]
                ds_id = ds_obj["track_id"] if ds_obj is not None else "None"
                gt_id = gt_obj["track_id"]
                if ds_obj is not None:
                    frame_ds = disp_track(frame, ds_obj)
                    predicted_x = np.append(predicted_x, float(ds_obj["x"]))
                    predicted_y = np.append(predicted_y, float(ds_obj["y"]))
                else:
                    predicted_x = np.append(predicted_x, np.nan)
                    predicted_y = np.append(predicted_y, np.nan)
                frame_gt = disp_track(frame, gt_obj)
                print(
                    f"frame: {frame_idx}, {gt_id}->{ds_id}, {gt_cls}->{ds_cls}, err: {err}"
                )
            else:
                expected_x = np.append(expected_x, np.nan)
                predicted_x = np.append(predicted_x, np.nan)
                expected_y = np.append(expected_y, np.nan)
                predicted_y = np.append(predicted_y, np.nan)

            post_data = {
                "expected_x": expected_x,
                "expected_y": expected_y,
                "predicted_x": predicted_x,
                "predicted_y": predicted_y,
                "time": time,
            }
            r = requests.post(url, data=post_data)
            # print result of request
            print(r)
            print(r.text)

            #         # update traces in plotly
            #         marker_data_pred.x = time
            #         marker_data_pred.y = predicted_x
            #         marker_data_pred.z = predicted_y
            #         marker_data_expected.x = time
            #         marker_data_expected.y = expected_x
            #         marker_data_expected.z = expected_y
            #         fig.update_traces(marker_data_pred, selector=dict(name="predicted"))
            #         fig.update_traces(marker_data_expected, selector=dict(name="expected"))

            #         #### SHOW RESULTS
            frame = np.concatenate((frame_ds, frame_gt), axis=0)
            cv2.imshow("Deepsort", frame)
            # break the loop if 'q' is pressed
            if cv2.waitKey(0) & 0xFF == ord("q"):
                break  ############ PLOT RESULTS

    # marker_data_pred = go.Scatter3d(
    #     x=time,
    #     y=predicted_x,
    #     z=predicted_y,
    #     marker=go.scatter3d.Marker(size=2),
    #     mode="markers",
    #     marker_color="red",
    # )
    # marker_data_expected = go.Scatter3d(
    #     x=time,
    #     y=expected_x,
    #     z=expected_y,
    #     marker=go.scatter3d.Marker(size=2),
    #     mode="markers",
    #     marker_color="blue",
    # )
    # # set trace names
    # marker_data_pred.name = "predicted"
    # marker_data_expected.name = "expected"
    # # fig = go.Figure(data=marker_data_expected)
    # fig = go.Figure(data=[marker_data_pred, marker_data_expected])
    # fig.update_layout(
    #     scene=dict(xaxis_title="time", yaxis_title="x", zaxis_title="y")
    # )
    # exist = []
    # fig.for_each_trace(lambda t: exist.append(t.name))
    # print(exist)
    # exit(0)

    for frame in time:
        gt_maps = ground_truth2deep_sort[frame]
        if GT_ID_OF_INTEREST in gt_maps:
            gt_obj = gt_maps[GT_ID_OF_INTEREST]["gt_obj"]
            expected_x = np.append(expected_x, float(gt_obj["x"]))
            expected_y = np.append(expected_y, float(gt_obj["y"]))
            if gt_maps[GT_ID_OF_INTEREST]["ds_obj"] is not None:
                ds_obj = gt_maps[GT_ID_OF_INTEREST]["ds_obj"]
                predicted_x = np.append(predicted_x, float(ds_obj["x"]))
                predicted_y = np.append(predicted_y, float(ds_obj["y"]))
            else:
                predicted_x = np.append(predicted_x, np.nan)
                predicted_y = np.append(predicted_y, np.nan)
        else:
            expected_x = np.append(expected_x, np.nan)
            predicted_x = np.append(predicted_x, np.nan)
            expected_y = np.append(expected_y, np.nan)
            predicted_y = np.append(predicted_y, np.nan)

    # do a 3D plotly plot of the predicted['x'], predicted['y'], time and actual['x'], actual['y'], time
    # marker_data_pred = go.Scatter3d(
    #     x=time,
    #     y=predicted_x,
    #     z=predicted_y,
    #     marker=go.scatter3d.Marker(size=2),
    #     mode="markers",
    #     marker_color="red",
    # )
    # marker_data_expected = go.Scatter3d(
    #     x=time,
    #     y=expected_x,
    #     z=expected_y,
    #     marker=go.scatter3d.Marker(size=2),
    #     mode="markers",
    #     marker_color="blue",
    # )
    # # fig = go.Figure(data=marker_data_expected)
    # fig = go.Figure(data=[marker_data_pred, marker_data_expected])
    # fig.update_layout(scene=dict(xaxis_title="time", yaxis_title="x", zaxis_title="y"))
    # fig.show()
