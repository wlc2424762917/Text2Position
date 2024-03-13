from typing import List

import os
import os.path as osp
import pickle
import numpy as np
import cv2
from easydict import EasyDict
from numpy.lib.function_base import flip

import torch
from torch.utils.data import Dataset, DataLoader

from torch_geometric.data import Data, Batch
import torch_geometric.transforms as T

from datapreparation.kitti360pose.utils import (
    CLASS_TO_LABEL,
    LABEL_TO_CLASS,
    CLASS_TO_MINPOINTS,
    CLASS_TO_INDEX,
    COLORS,
    COLOR_NAMES,
    SCENE_NAMES,
)
from datapreparation.kitti360pose.imports import Object3d, Cell, Pose
from datapreparation.kitti360pose.drawing import show_pptk, show_objects, plot_cell
from dataloading.kitti360pose.base import Kitti360BaseDataset
from dataloading.kitti360pose.utils import batch_object_points, flip_pose_in_cell
import yaml
from copy import deepcopy
import MinkowskiEngine as ME
import albumentations as A
from pathlib import Path


def get_all_points(objects):
    all_xyz = np.concatenate([obj.xyz for obj in objects], axis=0)
    all_rgb = np.concatenate([obj.rgb for obj in objects], axis=0)
    #
    all_semantic = np.array([CLASS_TO_INDEX[obj.label] for obj in objects for _ in obj.xyz])
    all_instance = np.array([obj.id for obj in objects for _ in obj.xyz])

    return all_xyz, all_rgb, all_semantic, all_instance

def load_yaml(filepath):
    with open(filepath) as f:
        # file = yaml.load(f, Loader=Loader)
        file = yaml.load(f)
    return file

def select_correct_labels(labels, num_labels):
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

def remap_from_zero(labels):
    labels_db = load_yaml(Path("/home/planner/wlc/Mask3D/data/processed/t2p/label_database.yaml"))
    label_info = select_correct_labels(labels_db, num_labels=22)
    labels[
        ~np.isin(labels, list(label_info.keys()))
    ] = 255
    # remap to the range from 0
    for i, k in enumerate(label_info.keys()):
        # print(f"k: {k}, i: {i}")
        labels[labels == k] = i
    return labels

def prepare_data(points, colors, all_semantic, all_instance, add_raw_coordinates=True):
    # combine all_semantic and all_instance to shape [n, 2]
    all_semantic = all_semantic[:, np.newaxis]  # [n, 1]
    all_instance = all_instance[:, np.newaxis]  # [n, 1]
    labels = np.concatenate((all_semantic, all_instance), axis=1)  # [n, 2]
    segments = np.ones((labels.shape[0], 1), dtype=np.float32)
    labels = labels.astype(np.int32)
    if labels.size > 0:
        labels[:, 0] = remap_from_zero(labels[:, 0])
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
    if add_raw_coordinates:
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

def load_pose_and_cell(pose: Pose, cell: Cell, hints, pad_size, transform, args, flip_pose=False):
    assert pose.cell_id == cell.id

    descriptions = pose.descriptions
    cell_objects_dict = {obj.id: obj for obj in cell.objects}
    matched_ids = [descr.object_id for descr in descriptions if descr.is_matched]
    matched_objects = [cell_objects_dict[matched_id] for matched_id in matched_ids]

    assert len(pose.descriptions) == args.num_mentioned
    assert len(pose.descriptions) - pose.get_number_unmatched() == len(matched_objects)

    # Hints and descriptions have to be in same order
    for descr, hint in zip(descriptions, hints):
        assert descr.object_label in hint

    # Gather offsets
    # NOTE: Currently trains on best-offsets if available (matched)!
    if args.regressor_cell == "pose" and args.regressor_learn == "closest":
        offsets = np.array([descr.offset_closest for descr in descriptions])[:, 0:2]
    if args.regressor_cell == "pose" and args.regressor_learn == "center":
        offsets = np.array([descr.offset_center for descr in descriptions])[:, 0:2]
    if args.regressor_cell == "best" and args.regressor_learn == "closest":
        # offsets = np.array([descr.best_offset_closest for descr in descriptions])[:, 0:2]
        offsets = []
        for i_descr, descr in enumerate(descriptions):
            if descr.is_matched:
                offsets.append(descr.best_offset_closest[0:2])
            else:
                offsets.append(descr.offset_closest[0:2])
    if args.regressor_cell == "best" and args.regressor_learn == "center":
        # offsets = np.array([descr.best_offset_center for descr in descriptions])[:, 0:2]
        offsets = []
        for i_descr, descr in enumerate(descriptions):
            if descr.is_matched:
                offsets.append(descr.best_offset_center[0:2])
            else:
                offsets.append(descr.offset_center[0:2])

    offsets = np.array(offsets)

    # Build best-center offsets for oracle
    offsets_best_center = []
    for i_descr, descr in enumerate(descriptions):
        if descr.is_matched:
            offsets_best_center.append(descr.best_offset_center[0:2])
        else:
            offsets_best_center.append(descr.offset_center[0:2])

    offsets_valid = np.sum(np.isnan(offsets)) == 0

    # Gather matched objects
    objects, matches = [], []  # Matches as [(obj_idx, hint_idx)]
    for i_descr, descr in enumerate(descriptions):
        if descr.is_matched:
            hint_obj = cell_objects_dict[descr.object_id]
            assert hint_obj.instance_id == descr.object_instance_id
            objects.append(hint_obj)

            obj_idx = len(objects) - 1
            hint_idx = i_descr
            matches.append((obj_idx, hint_idx))

    # Gather distractors, i.e. remaining objects
    for obj in cell.objects:
        if obj.id not in matched_ids:
            objects.append(obj)
    if len(objects) != len(cell.objects):
        print([obj.id for obj in objects])
        print([obj.id for obj in cell.objects])
        print(matched_ids)
    assert len(objects) == len(
        cell.objects
    ), f"Not all cell-objects have been gathered! {len(objects)}, {len(cell.objects)}, {cell.id}"

    # Pad or cut-off distractors (CARE: the latter would use ground-truth data!)
    if len(objects) > pad_size:
        objects = objects[0:pad_size]

    while len(objects) < pad_size:
        obj = Object3d.create_padding()
        objects.append(obj)

    # Build matches and all_matches
    # The matched objects are always put in first, however, our geometric models have no knowledge of these indices as they are permutation invariant.
    all_matches = matches.copy()

    # Add unmatched hints
    for i_descr, descr in enumerate(descriptions):
        if not descr.is_matched:
            obj_idx = len(objects)  # Match to objects-side bin
            hint_idx = i_descr
            all_matches.append((obj_idx, hint_idx))

    # Add unmatched objects
    for obj_idx, obj in enumerate(objects):
        if obj.id not in matched_ids:
            hint_idx = len(descriptions)  # Match to hints-side bin
            all_matches.append((obj_idx, hint_idx))

    matches, all_matches = np.array(matches), np.array(all_matches)
    assert len(matches) == len(matched_ids)
    assert len(all_matches) == len(objects) + len(descriptions) - len(matches)
    assert np.sum(all_matches[:, 1] == len(descriptions)) == len(objects) - len(
        matched_ids
    )  # Binned objects
    assert np.sum(all_matches[:, 0] == len(objects)) == len(descriptions) - len(
        matched_ids
    )  # Binned hints

    text = " ".join(hints)

    # TODO: flip here. This does not affect the the order and matches.
    # Flip objects, hint_descriptions, object_points, offsets, pose
    if flip_pose:
        if np.random.choice((True, False)):
            pose, cell, text, text_objects, text_submap, hints, offsets = flip_pose_in_cell(
                pose, cell, text, text, text,1, hints, offsets
            )  # Horizontal
        if np.random.choice((True, False)):
            pose, cell, text, text_objects, text_submap, hints, offsets = flip_pose_in_cell(
                pose, cell, text, text, text, -1, hints, offsets
            )  # Vertical

    object_points = batch_object_points(objects, transform)

    object_class_indices = [CLASS_TO_INDEX[obj.label] for obj in objects]
    object_color_indices = [COLOR_NAMES.index(obj.get_color_text()) for obj in objects]

    all_xyz, all_rgb, all_semantic, all_instance = get_all_points(cell.objects)
    coordinates, points, colors, features, unique_map, inverse_map, labels = prepare_data(all_xyz, all_rgb,
                                                                                          all_semantic,
                                                                                          all_instance)

    return {
        "poses": pose,
        "cells": cell,
        "objects": objects,
        "object_points": object_points,
        "hint_descriptions": hints,
        "texts": text,
        "matches": matches,
        "all_matches": all_matches,
        "offsets": np.array(offsets),
        "offsets_best_center": np.array(offsets_best_center),
        'offsets_valid': offsets_valid,
        "object_class_indices": object_class_indices,
        "object_color_indices": object_color_indices,

        "poses_coords": pose.pose,

        "cells_coordinates": coordinates,
        "cells_points": points,
        "cells_colors": colors,
        "cells_features": features,
        "cells_unique_map": unique_map,
        "cells_inverse_map": inverse_map,
        "cells_labels": labels,
    }


class Kitti360FineDataset(Kitti360BaseDataset):
    def __init__(self, base_path, scene_name, transform, args, flip_pose=False):
        """Dataset to train the fine module.

        Args:
            base_path: Data root path
            scene_name: Scene name
            transform: PyG transform on object points
            args: Global training arguments
            flip_pose (bool, optional): Flip poses to opposite site of the cell (including the hint direction). Defaults to False.
        """
        super().__init__(base_path, scene_name)
        self.pad_size = args.pad_size
        self.transform = transform
        self.flip_pose = flip_pose
        self.args = args

    def __getitem__(self, idx):
        pose = self.poses[idx]
        cell = self.cells_dict[pose.cell_id]
        hints = self.hint_descriptions[idx]

        return load_pose_and_cell(
            pose, cell, hints, self.pad_size, self.transform, self.args, flip_pose=self.flip_pose
        )

    def __len__(self):
        return len(self.poses)

    def collate_fn(data):
        batch = {}
        for key in data[0].keys():
            batch[key] = [data[i][key] for i in range(len(data))]
        return batch


class Kitti360FineDatasetMulti(Dataset):
    def __init__(self, base_path, scene_names, transform, args, flip_pose=False):
        """Multi-scene version of Kitti360FineDataset.

        Args:
            base_path: Data root path
            scene_name: Scene name
            transform: PyG transform on object points
            args: Global training arguments
            flip_pose (bool, optional): Flip poses to opposite site of the cell (including the hint direction). Defaults to False.
        """
        self.scene_names = scene_names
        self.flip_pose = flip_pose
        self.datasets = [
            Kitti360FineDataset(base_path, scene_name, transform, args, flip_pose)
            for scene_name in scene_names
        ]

        self.all_poses = [pose for dataset in self.datasets for pose in dataset.poses]  # For stats
        self.all_cells = [
            cell for dataset in self.datasets for cell in dataset.cells
        ]  # For eval stats

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

    def __repr__(self):
        poses = np.array([pose.pose_w for pose in self.all_poses])
        num_poses = len(
            np.unique(poses, axis=0)
        )  # CARE: Might be possible that is is slightly inaccurate if there are actually overlaps
        return f"Kitti360FineDatasetMulti: {len(self)} descriptions for {num_poses} unique poses from {len(self.datasets)} scenes, {len(self.all_cells)} cells, flip: {self.flip_pose}."

    def __len__(self):
        return np.sum([len(ds) for ds in self.datasets])

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


if __name__ == "__main__":
    base_path = "./data/k360_30-10_scG_pd10_pc4_spY_all"
    folder_name = "2013_05_28_drive_0003_sync"

    args = EasyDict(pad_size=8, num_mentioned=6, ranking_loss="pairwise", regressor_cell="pose", regressor_learn="center", regressor_eval="center")
    transform = T.Compose([T.FixedPoints(1024), T.NormalizeScale()])

    dataset = Kitti360FineDatasetMulti(
        base_path,
        [
            folder_name,
        ],
        transform,
        args,
    )
    data = dataset[0]
