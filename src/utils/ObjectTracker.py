#!/usr/bin/env python
# Created by Jonathan Mikler on 08/May/23

import pathlib
import numpy as np

from typing import Dict, List


class ObjectTracker:
    def __init__(self):
        self.objects_in_time: Dict[
            int, Dict
        ] = dict()  # time -> trajectory:= dict of obj.id -> np.ndarray of shape (3)

    def update_position(self, time_: int, obj_key_: int, position_: np.ndarray):
        """
        Registers the position of the object at time_
        args:
            time_: int      -- time step
            obj_key_: int   -- object id
            position_: np.ndarray -- position of the object at time_
        """
        assert position_.shape == (3,), "Position must be a np.ndarray of shape (3,)"

        print(f"ObjectTracker | Updating position of object {obj_key_} at time {time_}")
        if self.objects_in_time.get(time_):
            self.objects_in_time[time_][obj_key_] = position_
        else:
            self.objects_in_time[time_] = {obj_key_: position_}

    def predict_position(self, obj_key_: int, time_: int) -> np.ndarray:
        """
        Predicts the position of the object at before_time_ + 1
        args:
            obj_key_: int   -- object id
            time_: int      -- time step
        """
        try:
            p1 = self.objects_in_time[time_ - 1][obj_key_]
            p2 = self.objects_in_time[time_ - 2][obj_key_]
            dx = p1 - p2
            return self.objects_in_time[time_ - 1][obj_key_] + dx
            # dx = self.objects_in_time[time_-1][obj_key_] - self.objects_in_time[time_ - 2][obj_key_]
        except Exception as e:
            print(
                "Not possible to predict position of object with id: {} at time: {}".format(
                    obj_key_, time_
                )
            )
            return None

    def get_object_trajectory(self, obj_key_: int) -> List[np.ndarray]:
        """
        Return a list of positions of the object with obj_key_ given
        in the case of the object not having a registered position at a given time step,
        the position is None.
        """
        return [
            self.objects_in_time[time_].get(obj_key_, None)
            for time_ in self.objects_in_time.keys()
        ]


if __name__ == "__main__":
    pass
