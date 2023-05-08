#!/usr/bin/env python
# Created by Jonathan Mikler on 08/May/23

import pathlib
import numpy as np

from typing import Dict

class ObjectTracker():
    _trajectory: Dict[int, np.ndarray] # obj.id -> np.ndarray of shape (3) # could be a np.ndarray of shape (t,3)
    def __init__(self):
        self.objects_in_time: Dict[int, Dict] = dict()  # time -> trajectory:= dict of obj.id -> np.ndarray of shape (3)
        self.tracked_kinematics: Dict[int, np.ndarray] = dict() # obj.id -> deltaX # QUESTION: Maybe we want to keep track of the last K deltas?

    def update_position(self, time_:int, obj_key_:int, position_:np.ndarray):
        """
        Registers the position of the object at time_
        args:
            time_: int      -- time step
            obj_key_: int   -- object id
            position_: np.ndarray -- position of the object at time_
        """
        if obj_key_ in self.objects_in_time[time_].keys():
            self.objects_in_time[time_][obj_key_] = position_
        else:
            self.objects_in_time[time_] = {obj_key_: position_}
    def register_kinematics(self, obj_key_:int, time_:int, dx_:np.ndarray):
        if obj_key_ in self.tracked_kinematics.keys():
            self.tracked_kinematics[obj_key_][time_] = dx_
        else:
            self.tracked_kinematics[obj_key_] = {time_: dx_}
    
    def predict_position(self, obj_key_:int, before_time_:int) -> np.ndarray:
        """
        Predicts the position of the object at before_time_ + 1
        args:
            obj_key_: int   -- object id
            time_: int      -- time step
        """
        assert obj_key_ in self.objects_in_time.keys(), f"Object {obj_key_} not found in tracker"
        assert before_time_ in self.objects_in_time[obj_key_].keys(), f"Position of object {obj_key_} not found at time {before_time_}"
        assert before_time_ in self.tracked_kinematics[obj_key_].keys(), f"Kinematics of object {obj_key_} not found at time {before_time_}"

        return self.objects_in_time.get(obj_key_).get(before_time_) + self.tracked_kinematics.get(obj_key_).get(before_time_)

    def get_object_trajectory(self, obj_key_:int) -> Dict[int, np.ndarray]:
        return self.objects_in_time.get(obj_key_)

    def save_trajectories(self, save_dir_:pathlib.Path):
        """
        Saves the trajectories to a txt file
        args:
            save_dir_: pathlib.Path -- directory to save the trajectories to
        """
        _row = []


if __name__ == '__main__':
    pass