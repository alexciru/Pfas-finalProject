#!/usr/bin/env python
# Created by Jonathan Mikler on 08/May/23

import pathlib
import numpy as np

from typing import Dict

class ObjectTracker():
    _trajectory: Dict[int, np.ndarray] # obj.id -> np.ndarray of shape (3) # could be a np.ndarray of shape (t,3)
    def __init__(self):
        self.tracked_objects: Dict[int, Dict] = dict()  # obj.id -> trajectory
        self.tracked_kinematics: Dict[int, np.ndarray] = dict() # obj.id -> deltaX # QUESTION: Maybe we want to keep track of the last K deltas?

    def register_position(self, obj_key_:int, time_:int, position_:np.ndarray):
        if obj_key_ in self.tracked_objects.keys():
            self.tracked_objects[obj_key_][time_] = position_
        else:
            self.tracked_objects[obj_key_] = {time_: position_}

        if time_ > 0:
            self.register_kinematics(obj_key_, time_, position_ - self.tracked_objects[obj_key_][time_ - 1])

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
        assert obj_key_ in self.tracked_objects.keys(), f"Object {obj_key_} not found in tracker"
        assert before_time_ in self.tracked_objects[obj_key_].keys(), f"Position of object {obj_key_} not found at time {before_time_}"
        assert before_time_ in self.tracked_kinematics[obj_key_].keys(), f"Kinematics of object {obj_key_} not found at time {before_time_}"

        return self.tracked_objects.get(obj_key_).get(before_time_) + self.tracked_kinematics.get(obj_key_).get(before_time_)

    def get_object_trajectory(self, obj_key_:int) -> Dict[int, np.ndarray]:
        return self.tracked_objects.get(obj_key_)

    def save_trajectories(self, save_dir_:pathlib.Path):
        """
        Saves the trajectories to a txt file
        args:
            save_dir_: pathlib.Path -- directory to save the trajectories to
        """


if __name__ == '__main__':
    pass