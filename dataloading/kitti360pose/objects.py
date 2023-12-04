from typing import List

import os
import os.path as osp
import pickle
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset
import torch_geometric.transforms as T
from torch_geometric.data import Data, DataLoader

from datapreparation.kitti360pose.utils import (
    CLASS_TO_LABEL,
    LABEL_TO_CLASS,
    CLASS_TO_MINPOINTS,
    SCENE_NAMES,
    COLORS,
    COLOR_NAMES,
)
from datapreparation.kitti360pose.imports import Object3d, Cell
from datapreparation.kitti360pose.drawing import show_pptk, show_objects, plot_cell
from dataloading.kitti360pose.base import Kitti360BaseDataset


class Kitti360ObjectsDataset(Kitti360BaseDataset):
    def __init__(
        self, base_path, scene_name, transform=T.Compose([T.FixedPoints(1024), T.NormalizeScale()])
    ):
        """Dataset for Kitti360 object classification training.
        NOTE: should be shuffled so that objects aren't ordered by class.
        Objects will often have less than 2k points, T.FixedPoints() will sample w/ replace by default.

        Args:
            base_path: Data root path
            scene_name: Scene name
            transform (optional): PyG transform for object points. Defaults to T.Compose([T.FixedPoints(1024), T.NormalizeScale()]).
        """
        super().__init__(base_path, scene_name)
        self.transform = transform

        # Note: objects are retrieved from cells, not the (un-clustered) scene-objects
        self.objects = [obj for cell in self.cells for obj in cell.objects]

        # print(self)

    def __getitem__(self, idx):
        obj = self.objects[idx]
        points = torch.tensor(np.float32(obj.xyz))
        colors = torch.tensor(np.float32(obj.rgb))
        label = self.class_to_index[obj.label]
        label_color = COLOR_NAMES.index(obj.get_color_text())
        assert label_color >= 0 and label_color < 8

        data = Data(
            x=colors, y=label, pos=points, y_color=label_color
        )  # 'x' refers to point-attributes in PyG, 'pos' is xyz
        data = self.transform(data)
        return data

    def __len__(self):
        return len(self.objects)

    def __repr__(self):
        return (
            f"Kitti360ObjectsDataset: {len(self)} objects from {len(self.class_to_index)} classes"
        )


class Kitti360ObjectsDatasetMulti(Dataset):
    def __init__(
        self, base_path, scene_names, transform=T.Compose([T.FixedPoints(1024), T.NormalizeScale()])
    ):
        """Multi-scene version of Kitti360ObjectsDataset.
        NOTE: should be shuffled so that objects aren't ordered by class.
        Objects will often have less than 2k points, T.FixedPoints() will sample w/ replace by default.

        Args:
            base_path: Data root path
            scene_name: Scene name
            transform (optional): PyG transform for object points. Defaults to T.Compose([T.FixedPoints(1024), T.NormalizeScale()]).
        """
        self.scene_names = scene_names
        self.transform = transform
        self.datasets = [
            Kitti360ObjectsDataset(base_path, scene_name, transform) for scene_name in scene_names
        ]

        self.objects = [obj for dataset in self.datasets for obj in dataset.objects]
        self.class_to_index = self.datasets[0].class_to_index

        print(self)

    def __getitem__(self, idx):
        obj = self.objects[idx]
        points = torch.tensor(np.float32(obj.xyz))
        colors = torch.tensor(np.float32(obj.rgb))
        label = self.class_to_index[obj.label]
        label_color = COLOR_NAMES.index(obj.get_color_text())
        assert label_color >= 0 and label_color < 8

        data = Data(
            x=colors, y=label, pos=points, y_color=label_color
        )  # 'x' refers to point-attributes in PyG, 'pos' is xyz
        data = self.transform(data)
        return data

    def __len__(self):
        return len(self.objects)

    def __repr__(self):
        return f"Kitti360ObjectsDatasetMulti: {len(self.scene_names)} scenes, {len(self)} objects from {len(self.get_known_classes())} classes"

    def get_known_classes(self):
        return list(self.class_to_index.keys())


if __name__ == "__main__":
    base_path = "./data/k360_decouple"
    folder_name = "2013_05_28_drive_0003_sync"

    dataset = Kitti360ObjectsDatasetMulti(
        base_path,
        [
            folder_name,
        ],
    )
    data = dataset[0]
