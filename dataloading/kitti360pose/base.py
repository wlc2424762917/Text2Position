from typing import List

import os
import os.path as osp
import pickle
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset, DataLoader

from datapreparation.kitti360pose.utils import (
    CLASS_TO_LABEL,
    LABEL_TO_CLASS,
    CLASS_TO_MINPOINTS,
    SCENE_NAMES,
    CLASS_TO_INDEX,
)
from datapreparation.kitti360pose.imports import Object3d, Cell, Pose
from datapreparation.kitti360pose.drawing import (
    show_pptk,
    show_objects,
    plot_cell,
    plot_pose_in_best_cell,
)
import time


class Kitti360BaseDataset(Dataset):
    def __init__(self, base_path, scene_name):
        """Base dataset for loading Kitti360Pose data.

        Args:
            base_path: Base path for the Kitti360Pose scenes
            scene_name: Name of the scene to load
        """
        self.scene_name = scene_name
        self.cells = pickle.load(
            open(osp.join(base_path, "cells", f"{scene_name}.pkl"), "rb")
        )  # Also use objects from here for classification
        self.cells_dict = {cell.id: cell for cell in self.cells}  # For faster access

        cell_ids = [cell.id for cell in self.cells]
        assert len(np.unique(cell_ids)) == len(cell_ids)

        self.poses = pickle.load(open(osp.join(base_path, "poses", f"{scene_name}.pkl"), "rb"))  # target positions

        self.class_to_index = CLASS_TO_INDEX

        self.direction_map = {"on-top": "base", "north": "south", "south": "north", "east": "west", "west": "east"}

        self.hint_descriptions = [
            Kitti360BaseDataset.create_hint_description(pose, self.cells_dict[pose.cell_id])
            for pose in self.poses
        ]

        self.objects_descriptions = [
            Kitti360BaseDataset.create_objects_description(pose, self.cells_dict[pose.cell_id])
            for pose in self.poses
        ]

        self.submap_descriptions = [
            Kitti360BaseDataset.create_submap_description(self, pose, self.cells_dict[pose.cell_id])
            for pose in self.poses
        ]

    def __getitem__(self, idx):
        raise Exception("Not implemented: abstract class.")

    def create_hint_description(pose: Pose, cell: Cell):
        hints = []
        # cell_objects_dict = {obj.id: obj for obj in cell.objects}
        # time_start_create_hint_description = time.time()
        for descr in pose.descriptions:
            # obj = cell_objects_dict[descr.object_id]
            # hints.append(f'The pose is {descr.direction} of a {obj.get_color_text()} {obj.label}.')
            hints.append(
                f"The pose is {descr.direction} of a {descr.object_color_text} {descr.object_label}."
            )
        # time_end_create_hint_description = time.time()
        # print("time_create_hint_description: ", time_end_create_hint_description - time_start_create_hint_description)
        return hints

    def create_objects_description(pose: Pose, cell: Cell):
        hints = []
        # cell_objects_dict = {obj.id: obj for obj in cell.objects}
        for descr in pose.descriptions:
            # obj = cell_objects_dict[descr.object_id]
            # hints.append(f'The pose is {descr.direction} of a {obj.get_color_text()} {obj.label}.')
            hints.append(
                f"There is a {descr.object_color_text} {descr.object_label}."
            )
        return hints

    def create_submap_description(self, pose: Pose, cell: Cell):
        hints = []
        # cell_objects_dict = {obj.id: obj for obj in cell.objects}
        for descr in pose.descriptions:
            # obj = cell_objects_dict[descr.object_id]
            # hints.append(f'The pose is {descr.direction} of a {obj.get_color_text()} {obj.label}.')
            new_direction = self.direction_map[descr.direction]
            hints.append(
                f"The submap has a {descr.object_color_text} {descr.object_label} on {new_direction}."
            )
        return hints

    def get_known_classes(self):
        return list(self.class_to_index.keys())

    def get_known_words(self):
        words = []
        for hints in self.hint_descriptions:
            for hint in hints:
                words.extend(hint.replace(".", "").replace(",", "").lower().split())
        return list(np.unique(words))

    def __len__(self):
        raise Exception("Not implemented: abstract class.")

    def collate_fn(data):
        batch = {}
        for key in data[0].keys():
            batch[key] = [data[i][key] for i in range(len(data))]
        return batch


if __name__ == "__main__":
    base_path = "./data/k360_decouple"
    folder_name = "2013_05_28_drive_0003_sync"

    dataset = Kitti360BaseDataset(base_path, folder_name)
