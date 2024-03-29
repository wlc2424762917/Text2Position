from typing import List
import sys
sys.path.append('/home/wanglichao/Text2Pos-CVPR2022')
import os
import os.path as osp
import pickle
import numpy as np
import cv2
from copy import deepcopy
import time
import yaml

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

import torch_geometric.transforms as T

from datapreparation.kitti360pose.utils import (
    CLASS_TO_LABEL,
    LABEL_TO_CLASS,
    CLASS_TO_MINPOINTS,
    SCENE_NAMES,
    SCENE_NAMES_TEST,
    SCENE_NAMES_TRAIN,
    SCENE_NAMES_VAL,
)
from datapreparation.kitti360pose.utils import CLASS_TO_INDEX, COLOR_NAMES
from datapreparation.kitti360pose.imports import Object3d, Cell, Pose
from datapreparation.kitti360pose.drawing import (
    show_pptk,
    show_objects,
    plot_cell,
    plot_pose_in_best_cell,
)
from dataloading.kitti360pose.base import Kitti360BaseDataset
from dataloading.kitti360pose.utils import batch_object_points, flip_pose_in_cell
from torch_geometric.data import Data, Batch
import albumentations as A
import MinkowskiEngine as ME


class Kitti360CoarseDataset(Kitti360BaseDataset):
    def __init__(
        self,
        base_path,
        scene_name,
        transform,
        shuffle_hints=False,
        flip_poses=False,
        sample_close_cell=False,
    ):
        """Dataset variant for coarse module training.
        Returns one item per pose.

        Args:
            base_path: Base path of the Kitti360Poses data
            scene_name: Scene name
            transform: PyG transform to apply to object_points
            shuffle_hints (bool, optional): Shuffle the hints of a description. Defaults to False.
            flip_poses (bool, optional): Flip the poses inside the cell. NOTE: Might make hints inaccurate. Defaults to False.
            sample_close_cell (bool, optional): Sample any close-by cell per pose instead of the original one. Defaults to False.
        """
        super().__init__(base_path, scene_name)
        self.shuffle_hints = shuffle_hints
        self.transform = transform
        self.flip_poses = flip_poses

        self.sample_close_cell = sample_close_cell
        self.cell_centers = np.array([cell.get_center()[0:2] for cell in self.cells])

        self.add_raw_coordinates = True
        label_db_filepath = "/home/planner/wlc/Mask3D/data/processed/t2p/label_database.yaml"
        labels = self._load_yaml(Path(label_db_filepath))
        self._labels = self._select_correct_labels(labels, num_labels=22)
        self.ignore_label= 255

    def get_all_points(self, objects):
        all_xyz = np.concatenate([obj.xyz for obj in objects], axis=0)
        all_rgb = np.concatenate([obj.rgb for obj in objects], axis=0)
        #
        # for obj in objects:
        #     print(f"obj.label: {obj.label}, {CLASS_TO_INDEX[obj.label]}, obj.color: {obj.get_color_text()}; obj.center: {obj.get_center()}")
        all_semantic = np.array([CLASS_TO_INDEX[obj.label] for obj in objects for _ in obj.xyz])
        all_instance = np.array([obj.id for obj in objects for _ in obj.xyz])

        return all_xyz, all_rgb, all_semantic, all_instance

    def _load_yaml(self, filepath):
        with open(filepath) as f:
            # file = yaml.load(f, Loader=Loader)
            file = yaml.load(f)
        return file

    def _select_correct_labels(self, labels, num_labels):
        number_of_validation_labels = 0
        number_of_all_labels = 0
        for (
            k,
            v,
        ) in labels.items():
            number_of_all_labels += 1
            if v["validation"]:
                number_of_validation_labels += 1

        if num_labels == number_of_all_labels:
            return labels
        elif num_labels == number_of_validation_labels:
            valid_labels = dict()
            for (
                k,
                v,
            ) in labels.items():
                if v["validation"]:
                    valid_labels.update({k: v})
            return valid_labels
        else:
            msg = f"""not available number labels, select from:
            {number_of_validation_labels}, {number_of_all_labels}"""
            raise ValueError(msg)

    def _remap_from_zero(self, labels):
        labels[
            ~np.isin(labels, list(self.label_info.keys()))
        ] = self.ignore_label
        # remap to the range from 0
        for i, k in enumerate(self.label_info.keys()):
            # print(f"k: {k}, i: {i}")
            labels[labels == k] = i
        return labels

    @property
    def label_info(self):
        """database file containing information labels used by dataset"""
        return self._labels

    def prepare_data(self, points, colors, all_semantic, all_instance):
        # combine all_semantic and all_instance to shape [n, 2]
        all_semantic = all_semantic[:, np.newaxis]  # [n, 1]
        all_instance = all_instance[:, np.newaxis]  # [n, 1]
        labels = np.concatenate((all_semantic, all_instance), axis=1)  # [n, 2]
        segments = np.ones((labels.shape[0], 1), dtype=np.float32)
        labels = labels.astype(np.int32)
        if labels.size > 0:
            labels[:, 0] = self._remap_from_zero(labels[:, 0])
            # always add instance
            # if not self.add_instance:
            #     # taking only first column, which is segmentation label, not instance
            #     labels = labels[:, 0].flatten()[..., None]

        labels = np.hstack((labels, segments[...].astype(np.int32)))
        # normalization for point cloud features
        points[:, 0:3] = points[:, 0:3] * 30

        # TODO: before centering or not
        points_ori = deepcopy(points)
        points[:, :3] = points[:, :3] - points[:, :3].min(0)

        # print(f"points min: {points.min(0)}")
        # print(f"points mean: {points.mean(0)}")
        points -= points.mean(0)  # center

        color_mean = (0.3003134802903658, 0.30814036261976874, 0.2635079033375686)
        color_std = (0.2619693782684099, 0.2592597800138297, 0.2448327959589299)
        normalize_color = A.Normalize(mean=color_mean, std=color_std)

        colors_ori = deepcopy(colors)
        colors = colors * 255.

        pseudo_image = colors.astype(np.uint8)[np.newaxis, :, :]
        colors = np.squeeze(normalize_color(image=pseudo_image)["image"])

        features = colors
        if self.add_raw_coordinates:
            features = np.hstack((features, points))  # last three columns are coordinates

        coords = np.floor(points / 0.15)
        _, _, unique_map, inverse_map = ME.utils.sparse_quantize(
            coordinates=coords,
            features=features,
            return_index=True,
            return_inverse=True,
        )
        # print(f"coords: {coords.shape}")
        # print(f"unique_map: {unique_map.shape}")
        # print(f"unique_map: {unique_map}")
        # print(f"inverse_map: {inverse_map.shape}")
        # print(f"inverse_map: {inverse_map}")
        sample_coordinates = coords[unique_map]
        sample_coordinates = torch.from_numpy(sample_coordinates).int()
        sample_features = features[unique_map]
        sample_features = torch.from_numpy(sample_features).float()
        sample_labels = labels[unique_map]
        sample_labels = torch.from_numpy(sample_labels).long()
        #print(f"sample_coordinates: {sample_coordinates.shape}")
        # quit()
        return sample_coordinates, points_ori, colors_ori, sample_features, unique_map, inverse_map, sample_labels

    def __getitem__(self, idx):

        # time_start_get_cell = time.time()
        pose = self.poses[idx]

        # TODO/NOTE: If it doesn't work, check if there is a problem with flipping later on
        if self.sample_close_cell:  # Sample cells where the pose is close to the center (less than half a cell size)
            cell_size = self.cells[0].cell_size
            dists = np.linalg.norm(self.cell_centers - pose.pose_w[0:2], axis=1)
            indices = np.argwhere(dists <= cell_size / 2).flatten()
            cell = self.cells[np.random.choice(indices)]
            assert np.linalg.norm(cell.get_center()[0:2] - pose.pose_w[0:2]) < cell_size
        else:
            cell = self.cells_dict[pose.cell_id]
        # time_end_get_cell = time.time()
        # print(f"time get cell = {time_end_get_cell-time_start_get_cell:0.2f}")

        # time_start_create_hints = time.time()
        hints = self.hint_descriptions[idx]
        objects_descriptions = self.objects_descriptions[idx]
        submap_descriptions = self.submap_descriptions[idx]
        # time_end_create_hints = time.time()
        # print(f"time create hints = {time_end_create_hints-time_start_create_hints:0.2f}")

        # time_start_shuffle_hints = time.time()
        if self.shuffle_hints:
            hints = np.random.choice(hints, size=len(hints), replace=False)
            objects_descriptions = np.random.choice(
                objects_descriptions, size=len(objects_descriptions), replace=False
            )
            submap_descriptions = np.random.choice(
                submap_descriptions, size=len(submap_descriptions), replace=False
            )
        # time_end_shuffle_hints = time.time()
        # print(f"time shuffle hints = {time_end_shuffle_hints - time_start_shuffle_hints:0.2f}")

        text = " ".join(hints)
        text_objects = " ".join(objects_descriptions)
        text_submap = " ".join(submap_descriptions)

        # time_start_flip_hints = time.time()
        # NOTE: hints are currently not flipped! (Only the text.)
        if self.flip_poses:
            if np.random.choice((True, False)):  # Horizontal
                pose, cell, text, text_objects, text_submap = flip_pose_in_cell(pose, cell, text, text_objects, text_submap, 1)
            if np.random.choice((True, False)):  # Vertical
                pose, cell, text, text_objects, text_submap = flip_pose_in_cell(pose, cell, text, text_objects, text_submap, direction=-1)
        # time_end_flip_hints = time.time()
        # print(f"time flip hints = {time_end_flip_hints - time_start_flip_hints:0.2f}")

        # time_start_batch_obj = time.time()
        all_xyz, all_rgb, all_semantic, all_instance = self.get_all_points(cell.objects)
        coordinates, points, colors, features, unique_map, inverse_map, labels = self.prepare_data(all_xyz, all_rgb, all_semantic, all_instance)
        # print(f"features: {features.shape}")
        # time_start_indice_obj = time.time()
        object_class_indices = [CLASS_TO_INDEX[obj.label] for obj in cell.objects]
        object_color_indices = [COLOR_NAMES.index(obj.get_color_text()) for obj in cell.objects]
        # time_end_indice_obj = time.time()
        # print(f"time indice obj = {time_end_indice_obj - time_start_indice_obj:0.2f}")

        return {
            "poses": pose,
            "cells": cell,
            "objects": cell.objects,
            "cells_all_xyz": all_xyz,
            "cells_all_rgb": all_rgb,

            "cells_coordinates": coordinates,
            "cells_points": points,
            "cells_colors": colors,
            "cells_features": features,
            "cells_unique_map": unique_map,
            "cells_inverse_map": inverse_map,
            "cells_labels": labels,

            "texts": text,
            "texts_objects": text_objects,
            "texts_submap": text_submap,
            "cell_ids": pose.cell_id,
            "scene_names": self.scene_name,
            "object_class_indices": object_class_indices,
            "object_color_indices": object_color_indices,
            "debug_hint_descriptions": hints,  # Care: Not shuffled etc!
        }

    def __len__(self):
        return len(self.poses)


class Kitti360CoarseDatasetMulti(Dataset):
    def __init__(
        self,
        base_path,
        scene_names,
        transform,
        shuffle_hints=False,
        flip_poses=False,
        sample_close_cell=False,
    ):
        """Multi-scene variant of Kitti360CoarseDataset.

        Args:
            base_path: Base path of the Kitti360Poses data
            scene_names: List of scene names
            transform: PyG transform to apply to object_points
            shuffle_hints (bool, optional): Shuffle the hints of a description. Defaults to False.
            flip_poses (bool, optional): Flip the poses inside the cell. NOTE: Might make hints inaccurate. Defaults to False.
            sample_close_cell (bool, optional): Sample any close-by cell per pose instead of the original one. Defaults to False.
        """
        self.scene_names = scene_names
        self.transform = transform
        self.flip_poses = flip_poses
        self.sample_close_cell = sample_close_cell
        self.datasets = [
            Kitti360CoarseDataset(
                base_path, scene_name, transform, shuffle_hints, flip_poses, sample_close_cell
            )
            for scene_name in scene_names
        ]

        self.all_cells = [
            cell for dataset in self.datasets for cell in dataset.cells
        ]  # For cell-only dataset
        self.all_poses = [pose for dataset in self.datasets for pose in dataset.poses]  # For eval

        cell_ids = [cell.id for cell in self.all_cells]
        assert len(np.unique(cell_ids)) == len(self.all_cells)  # IDs should not repeat

        print(str(self))

    def __getitem__(self, idx):
        count = 0
        for dataset in self.datasets:
            idx_in_dataset = idx - count
            if idx_in_dataset < len(dataset):
                return dataset[idx_in_dataset]
            else:
                count += len(dataset)
        assert False

    def __len__(self):
        return np.sum([len(ds) for ds in self.datasets])

    def __repr__(self):
        poses = np.array([pose.pose_w for pose in self.all_poses])
        num_poses = len(
            np.unique(poses, axis=0)
        )  # CARE: Might be possible that is is slightly inaccurate if there are actually overlaps
        return f"Kitti360CellDatasetMulti: {len(self.scene_names)} scenes, {len(self)} descriptions for {num_poses} unique poses, {len(self.all_cells)} cells, flip {self.flip_poses}, close-cell {self.sample_close_cell}"

    def get_known_words(self):
        known_words = []
        for ds in self.datasets:
            known_words.extend(ds.get_known_words())
        return list(np.unique(known_words))

    def get_known_classes(self):
        known_classes = []
        for ds in self.datasets:
            known_classes.extend(ds.get_known_classes())
        return list(np.unique(known_classes))

    def get_cell_dataset(self):
        return Kitti360CoarseCellOnlyDataset(self.all_cells, self.transform)


class Kitti360CoarseCellOnlyDataset(Dataset):
    def __init__(self, cells: List[Cell], transform):
        """Dataset to return only the cells for encoding during evaluation
        NOTE: The way the cells are read from the Cells-Only-Dataset, they may have been augmented differently during the actual training. Cells-Only does not flip and shuffle!
        """
        super().__init__()

        self.cells = cells
        self.transform = transform
        self.add_raw_coordinates = True
        label_db_filepath = "/home/planner/wlc/Mask3D/data/processed/t2p/label_database.yaml"
        labels = self._load_yaml(Path(label_db_filepath))
        self._labels = self._select_correct_labels(labels, num_labels=22)
        self.ignore_label= 255

    def get_all_points(self, objects):
        all_xyz = np.concatenate([obj.xyz for obj in objects], axis=0)
        all_rgb = np.concatenate([obj.rgb for obj in objects], axis=0)
        #
        all_semantic = np.array([CLASS_TO_INDEX[obj.label] for obj in objects for _ in obj.xyz])
        all_instance = np.array([obj.id for obj in objects for _ in obj.xyz])

        clip_feature_center_pair = {}
        for obj in objects:
            obj_center = obj.get_center()
            obj_clip_feature = obj.feature_2d
            clip_feature_center_pair[obj_center] = obj_clip_feature

        return all_xyz, all_rgb, all_semantic, all_instance, clip_feature_center_pair

    def _select_correct_labels(self, labels, num_labels):
        number_of_validation_labels = 0
        number_of_all_labels = 0
        for (
            k,
            v,
        ) in labels.items():
            number_of_all_labels += 1
            if v["validation"]:
                number_of_validation_labels += 1

        if num_labels == number_of_all_labels:
            return labels
        elif num_labels == number_of_validation_labels:
            valid_labels = dict()
            for (
                k,
                v,
            ) in labels.items():
                if v["validation"]:
                    valid_labels.update({k: v})
            return valid_labels
        else:
            msg = f"""not available number labels, select from:
            {number_of_validation_labels}, {number_of_all_labels}"""
            raise ValueError(msg)

    def _load_yaml(self, filepath):
        with open(filepath) as f:
            # file = yaml.load(f, Loader=Loader)
            file = yaml.load(f)
        return file

    def _remap_from_zero(self, labels):
        labels[
            ~np.isin(labels, list(self.label_info.keys()))
        ] = self.ignore_label
        # remap to the range from 0
        for i, k in enumerate(self.label_info.keys()):
            labels[labels == k] = i
        return labels

    @property
    def label_info(self):
        """database file containing information labels used by dataset"""
        return self._labels

    def prepare_data(self, points, colors, all_semantic, all_instance, clip_feature_center_pairs=None):
        # combine all_semantic and all_instance to shape [n, 2]
        all_semantic = all_semantic[:, np.newaxis]  # [n, 1]
        all_instance = all_instance[:, np.newaxis]  # [n, 1]
        labels = np.concatenate((all_semantic, all_instance), axis=1)  # [n, 2]
        segments = np.ones((labels.shape[0], 1), dtype=np.float32)
        labels = labels.astype(np.int32)
        if labels.size > 0:
            labels[:, 0] = self._remap_from_zero(labels[:, 0])
            # always add instance
            # if not self.add_instance:
            #     # taking only first column, which is segmentation label, not instance
            #     labels = labels[:, 0].flatten()[..., None]

        labels = np.hstack((labels, segments[...].astype(np.int32)))
        # normalization for point cloud features
        points[:, 0:3] = points[:, 0:3] * 30

        # TODO: before centering or not
        points_ori = deepcopy(points)
        points[:, :3] = points[:, :3] - points[:, :3].min(0)
        points -= points.mean(0)  # center

        color_mean = (0.3003134802903658, 0.30814036261976874, 0.2635079033375686)
        color_std = (0.2619693782684099, 0.2592597800138297, 0.2448327959589299)
        normalize_color = A.Normalize(mean=color_mean, std=color_std)

        colors_ori = deepcopy(colors)
        colors = colors * 255.

        pseudo_image = colors.astype(np.uint8)[np.newaxis, :, :]
        colors = np.squeeze(normalize_color(image=pseudo_image)["image"])

        features = colors
        if self.add_raw_coordinates:
            features = np.hstack((features, points))  # last three columns are coordinates

        coords = np.floor(points / 0.15)
        _, _, unique_map, inverse_map = ME.utils.sparse_quantize(
            coordinates=coords,
            features=features,
            return_index=True,
            return_inverse=True,
        )
        # print(f"coords: {coords.shape}")
        # print(f"unique_map: {unique_map.shape}")
        # print(f"unique_map: {unique_map}")
        # print(f"inverse_map: {inverse_map.shape}")
        # print(f"inverse_map: {inverse_map}")
        sample_coordinates = coords[unique_map]
        sample_coordinates = torch.from_numpy(sample_coordinates).int()
        sample_features = features[unique_map]
        sample_features = torch.from_numpy(sample_features).float()
        sample_labels = labels[unique_map]
        sample_labels = torch.from_numpy(sample_labels).long()
        # print(f"sample_coordinates: {sample_coordinates.shape}")
        # quit()
        return sample_coordinates, points_ori, colors_ori, sample_features, unique_map, inverse_map, sample_labels


    def __getitem__(self, idx):
        cell = self.cells[idx]
        assert len(cell.objects) >= 1
        object_points = batch_object_points(cell.objects, self.transform)

        all_xyz, all_rgb, all_semantic, all_instance, center_clip_feature_pairs = self.get_all_points(cell.objects)
        coordinates, points, colors, features, unique_map, inverse_map, labels = self.prepare_data(all_xyz, all_rgb, all_semantic, all_instance)
        return {
            "cells": cell,
            "cell_ids": cell.id,
            "objects": cell.objects,
            "object_points": object_points,

            "cells_coordinates": coordinates,
            "cells_points": points,
            "cells_colors": colors,
            "cells_features": features,
            "cells_unique_map": unique_map,
            "cells_inverse_map": inverse_map,
            "cells_labels": labels

        }

    def __len__(self):
        return len(self.cells)


if __name__ == "__main__":
    # base_path = "/home/wanglichao/Text2Position/data/k360_30-10_scG_pd10_pc4_spY_all/"
    base_path = "/home/wanglichao/Text2Pos-CVPR2022/data/k360_30-10_scG_pd10_pc4_spY_all/"
    transform = T.FixedPoints(256)

    for scene_names in (SCENE_NAMES, SCENE_NAMES_TRAIN, SCENE_NAMES_VAL, SCENE_NAMES_TEST):
        dataset = Kitti360CoarseDatasetMulti(
            base_path, scene_names, transform, shuffle_hints=False, flip_poses=False
        )
        data = dataset[0]
        for key, value in data.items():
            print(key)
        # pose, cell, cell_points, text = data['poses'], data['cells'], data['cells_points'], data['texts']
        # print(cell_points.shape)
        # offsets = np.array([descr.offset_closest for descr in pose.descriptions])
        # hints = text.split('.')
        # pose_f, cell_f, text_f, hints_f, offsets_f = flip_pose_in_cell(pose, cell, text, 1, hints=hints, offsets=offsets)

        # Gather information about duplicate descriptions
        # descriptors = []
        # for pose in dataset.all_poses:
        #     mentioned = sorted(
        #         [f"{d.object_label}_{d.object_color_text}_{d.direction}" for d in pose.descriptions]
        #     )
        #     descriptors.append(mentioned)
        #
        # unique, counts = np.unique(descriptors, return_counts=True)
        # # for d in descriptors[0:10]:
        # #     print('\t',d)
        # print(
        #     f"{len(descriptors)} poses, {len(unique)} uniques, {np.max(counts)} max duplicates, {np.mean(counts):0.2f} mean duplicates"
        # )
        # print("---- \n\n")