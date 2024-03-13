from typing import List

import os
import os.path as osp
import pickle
import numpy as np
import cv2
from copy import deepcopy

import torch
from torch.utils.data import Dataset, DataLoader
from easydict import EasyDict

import torch_geometric.transforms as T

from datapreparation.kitti360pose.imports import Object3d, Cell, Pose
from datapreparation.kitti360pose.drawing import (
    show_pptk,
    show_objects,
    plot_cell,
    plot_pose_in_best_cell,
)
from datapreparation.kitti360pose.utils import (
    CLASS_TO_LABEL,
    LABEL_TO_CLASS,
    CLASS_TO_MINPOINTS,
    CLASS_TO_INDEX,
    COLORS,
    COLOR_NAMES,
    SCENE_NAMES,
)
from dataloading.kitti360pose.poses import batch_object_points
from dataloading.kitti360pose.base import Kitti360BaseDataset
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

def get_instance_masks(
    list_labels,
    task,
    list_segments=None,
    ignore_class_threshold=100,
    filter_out_classes=[],
    label_offset=0,
):
    target = []
    for batch_id in range(len(list_labels)):
        label_ids = []
        masks = []
        segment_masks = []
        instance_ids = list_labels[batch_id][:, 1].unique()

        for instance_id in instance_ids:
            if instance_id == -1:
                continue

            # TODO is it possible that a ignore class (255) is an instance???
            # instance == -1 ???
            tmp = list_labels[batch_id][
                list_labels[batch_id][:, 1] == instance_id
            ]
            label_id = tmp[0, 0]

            if (
                label_id in filter_out_classes
            ):  # floor, wall, undefined==255 is not included
                continue

            if (
                255 in filter_out_classes
                and label_id.item() == 255
                and tmp.shape[0] < ignore_class_threshold
            ):
                continue

            label_ids.append(label_id)
            masks.append(list_labels[batch_id][:, 1] == instance_id)

            if list_segments:
                segment_mask = torch.zeros(
                    list_segments[batch_id].shape[0]
                ).bool()
                segment_mask[
                    list_labels[batch_id][
                        list_labels[batch_id][:, 1] == instance_id
                    ][:, 2].unique()
                ] = True
                segment_masks.append(segment_mask)

        if len(label_ids) == 0:
            return list()

        label_ids = torch.stack(label_ids)
        masks = torch.stack(masks)
        if list_segments:
            segment_masks = torch.stack(segment_masks)

        if task == "semantic_segmentation":
            new_label_ids = []
            new_masks = []
            new_segment_masks = []
            for label_id in label_ids.unique():
                masking = label_ids == label_id

                new_label_ids.append(label_id)
                new_masks.append(masks[masking, :].sum(dim=0).bool())

                if list_segments:
                    new_segment_masks.append(
                        segment_masks[masking, :].sum(dim=0).bool()
                    )

            label_ids = torch.stack(new_label_ids)
            masks = torch.stack(new_masks)

            if list_segments:
                segment_masks = torch.stack(new_segment_masks)

                target.append(
                    {
                        "labels": label_ids,
                        "masks": masks,
                        "segment_mask": segment_masks,
                    }
                )
            else:
                target.append({"labels": label_ids, "masks": masks})
        else:
            l = torch.clamp(label_ids - label_offset, min=0)

            if list_segments:
                target.append(
                    {
                        "labels": l,
                        "masks": masks,
                        "segment_mask": segment_masks,
                    }
                )
            else:
                target.append({"labels": l, "masks": masks})
    return target


class Kitti360FineEvalDataset(Dataset):
    def __init__(self, poses: List[Pose], cells: List[Cell], transform, args):
        """Dataset to evaluate the fine module in isolation.
        Needed to include recall, precision and offset accuracy metrics.

        Args:
            poses (List[Pose]): List of poses
            cells (List[Cell]): List of cells
            transform: PyG transform to apply to object points
            args: Global script arguments for evaluation
        """
        super().__init__()
        self.poses = poses
        self.transform = transform
        self.args = args

        self.cells_dict = {cell.id: cell for cell in cells}

        print(
            f"Kitti360FineEvalDataset: {len(self)} poses, {len(cells)} cells, pad {args.pad_size}"
        )

    def load_pose_and_cell(self, pose: Pose, cell: Cell):
        assert pose.cell_id == cell.id
        assert len(pose.descriptions) == self.args.num_mentioned

        # Padded version here
        matched_ids = []
        for descr in pose.descriptions:
            matched_ids.append(descr.object_id if descr.is_matched else None)

        cell_objects_dict = {obj.id: obj for obj in cell.objects}

        # Gather offsets for oracle
        pose_in_cell = (pose.pose_w - cell.bbox_w[0:3])[0:2] / cell.cell_size
        oracle_offsets = []
        for descr in pose.descriptions:
            if descr.is_matched:
                # oracle_offsets.append(descr.best_offset_center[0:2])
                obj = cell_objects_dict[descr.object_id]
                oracle_offsets.append(pose_in_cell - obj.get_center()[0:2])
            else:
                oracle_offsets.append(descr.offset_center)

        # Gather the objects and matches
        objects = []
        matches = []
        for obj_idx, obj in enumerate(cell.objects):
            objects.append(obj)
            if obj.id in matched_ids:
                hint_idx = matched_ids.index(obj.id)
                matches.append((obj_idx, hint_idx))

            if len(objects) >= self.args.pad_size:
                break

        # Pad if needed
        while len(objects) < self.args.pad_size:
            objects.append(Object3d.create_padding())
        assert len(objects) == self.args.pad_size

        matches = np.array(matches)
        assert len(matches) <= len(matched_ids)  # Some matched objects can be cut-off

        all_xyz, all_rgb, all_semantic, all_instance = get_all_points(cell.objects)
        coordinates, points, colors, features, unique_map, inverse_map, labels = prepare_data(all_xyz, all_rgb,
                                                                                              all_semantic,
                                                                                              all_instance)

        return {
            "poses": pose,
            "cells": cell,
            "objects": objects,
            "object_points": batch_object_points(objects, self.transform),
            "matches": matches,
            "hint_descriptions": Kitti360BaseDataset.create_hint_description(pose, None),
            "offsets_best_center": np.array(oracle_offsets),
            "poses_coords": pose.pose,

            "cells_coordinates": coordinates,
            "cells_points": points,
            "cells_colors": colors,
            "cells_features": features,
            "cells_unique_map": unique_map,
            "cells_inverse_map": inverse_map,
            "cells_labels": labels,
        }

    def __getitem__(self, idx: int):
        pose = self.poses[idx]
        cell = self.cells_dict[pose.cell_id]

        return self.load_pose_and_cell(pose, cell)

    def __len__(self):
        return len(self.poses)

    def collate_fn(data):
        batch = {}
        for key in data[0].keys():
            batch[key] = [data[i][key] for i in range(len(data))]
        return batch


class Kitti360TopKDataset(Dataset):
    def __init__(self, poses: List[Pose], cells: List[Cell], retrievals, transform, args):
        """Dataset to rune the fine module on one query against multiple cells.
        Return a "batch" of each pose with each of the corresponding top-k retrieved cells.

        Args:
            poses (List[Pose]): List of poses
            cells (List[Cell]): List of cells
            retrievals: List of lists of retrievals: [[cell_id_0, cell_id_1, ...], ...]
            transform: PyG transform for object points
            args: Global evaluation arguments
        """
        super().__init__()
        self.poses = poses
        self.retrievals = retrievals
        assert len(poses) == len(retrievals)
        assert len(retrievals[0]) == max(args.top_k), "Retrievals where not trimmed to max(top_k)"
        assert len(poses) != len(cells)

        self.cells_dict = {cell.id: cell for cell in cells}
        assert len(self.cells_dict) == len(cells), "Cell-IDs are not unique"

        self.transform = transform
        self.args = args

        print(
            f"Kitti360TopKDataset: {len(self.poses)} poses, {len(cells)} cells, pad {args.pad_size}"
        )

    def load_pose_and_cell(self, pose: Pose, cell: Cell):
        assert pose.cell_id == cell.id
        assert len(pose.descriptions) == self.args.num_mentioned

        # Padded version here
        matched_ids = []
        for descr in pose.descriptions:
            matched_ids.append(descr.object_id if descr.is_matched else None)

        cell_objects_dict = {obj.id: obj for obj in cell.objects}

        # Gather offsets for oracle
        pose_in_cell = (pose.pose_w - cell.bbox_w[0:3])[0:2] / cell.cell_size
        oracle_offsets = []
        for descr in pose.descriptions:
            if descr.is_matched:
                # oracle_offsets.append(descr.best_offset_center[0:2])
                obj = cell_objects_dict[descr.object_id]
                oracle_offsets.append(pose_in_cell - obj.get_center()[0:2])
            else:
                oracle_offsets.append(descr.offset_center)

        # Gather the objects and matches
        objects = []
        matches = []
        for obj_idx, obj in enumerate(cell.objects):
            objects.append(obj)
            if obj.id in matched_ids:
                hint_idx = matched_ids.index(obj.id)
                matches.append((obj_idx, hint_idx))

            if len(objects) >= self.args.pad_size:
                break

        # Pad if needed
        while len(objects) < self.args.pad_size:
            objects.append(Object3d.create_padding())
        assert len(objects) == self.args.pad_size

        matches = np.array(matches)
        assert len(matches) <= len(matched_ids)  # Some matched objects can be cut-off

        all_xyz, all_rgb, all_semantic, all_instance = get_all_points(cell.objects)
        coordinates, points, colors, features, unique_map, inverse_map, labels = prepare_data(all_xyz, all_rgb,
                                                                                              all_semantic,
                                                                                              all_instance)

        return {
            "poses": pose,
            "cells": cell,
            "objects": objects,
            "object_points": batch_object_points(objects, self.transform),
            "matches": matches,
            "hint_descriptions": Kitti360BaseDataset.create_hint_description(pose, None),
            "offsets_best_center": np.array(oracle_offsets),
            "poses_coords": pose.pose,

            "cells_coordinates": coordinates,
            "cells_points": points,
            "cells_colors": colors,
            "cells_features": features,
            "cells_unique_map": unique_map,
            "cells_inverse_map": inverse_map,
            "cells_labels": labels,
        }

    def __getitem__(self, idx):
        """Return a "batch" of the pose at idx with each of the corresponding top-k retrieved cells"""
        pose = self.poses[idx]
        retrievals = self.retrievals[idx]

        return Kitti360TopKDataset.collate_append(
            [self.collate_fn_cell(pose, self.cells_dict[cell_id]) for cell_id in retrievals]
        )

    # NOTE: returns the number of poses, each item has max(top_k) samples!
    def __len__(self):
        return len(self.poses)

    def collate_fn_cell(data):
        batch = {}
        for key in data[0].keys():
            batch[key] = [data[i][key] for i in range(len(data))]
        # print(f"batch['cells_coor']: {batch['cells_coordinates'][0].shape}")
        input_dict = {"coords": batch["cells_coordinates"], "feats": batch["cells_features"]}
        if len(batch["cells_labels"]) > 0:
            input_dict["labels"] = batch["cells_labels"]
            # for i in range(len(input_dict["labels"])):
            #     # change np to torch long
            #     input_dict["labels"][i] = torch.from_numpy(input_dict["labels"][i]).long()
            coordinates, features, labels = ME.utils.sparse_collate(**input_dict)
        else:
            coordinates, features = ME.utils.sparse_collate(**input_dict)
            labels = torch.Tensor([])

        input_dict["segment2label"] = []
        target = []

        if "labels" in input_dict:
            for i in range(len(input_dict["labels"])):
                # TODO BIGGER CHANGE CHECK!!!
                _, ret_index, ret_inv = np.unique(
                    input_dict["labels"][i][:, -1],
                    return_index=True,
                    return_inverse=True,
                )
                input_dict["labels"][i][:, -1] = torch.from_numpy(ret_inv)
                input_dict["segment2label"].append(
                    input_dict["labels"][i][ret_index][:, :-1]
                )

            list_labels = input_dict["labels"]

            if len(list_labels[0].shape) == 1:  # 1D labels
                for batch_id in range(len(list_labels)):
                    label_ids = list_labels[batch_id].unique()
                    if 255 in label_ids:
                        label_ids = label_ids[:-1]

                    target.append(
                        {
                            "labels": label_ids,
                            "masks": list_labels[batch_id]
                                     == label_ids.unsqueeze(1),
                        }
                    )
            else:
                target = get_instance_masks(
                    list_labels,
                    list_segments=input_dict["segment2label"],
                    task="instance_segmentation",
                    ignore_class_threshold=100,
                    filter_out_classes=[],
                    label_offset=1,
                )
                for i in range(len(target)):
                    target[i]["point2segment"] = input_dict["labels"][i][:, 2]

        batch["cells_coordinates"] = coordinates
        batch["cells_features"] = features
        batch["cells_labels"] = labels
        batch["cells_targets"] = target

        return batch


if __name__ == "__main__":
    from dataloading.kitti360pose.cells import Kitti360CoarseDatasetMulti

    base_path = "./data/k360_cs30_cd15_scY_pd10_pc1_spY_closest"
    folder_name = "2013_05_28_drive_0003_sync"

    args = EasyDict(pad_size=16, top_k=(1, 3, 5))

    transform = T.FixedPoints(256)
    dataset_coarse = Kitti360CoarseDatasetMulti(
        base_path,
        [
            folder_name,
        ],
        transform,
        shuffle_hints=False,
        flip_poses=False,
    )

    retrievals = []
    for i in range(len(dataset_coarse.all_poses)):
        retrievals.append([dataset_coarse.all_cells[k].id for k in range(max(args.top_k))])

    dataset = Kitti360TopKDataset(
        dataset_coarse.all_poses, dataset_coarse.all_cells, retrievals, transform, args
    )
    data = dataset[0]

    loader = DataLoader(dataset, batch_size=2, collate_fn=Kitti360TopKDataset.collate_append)
    batch = next(iter(loader))
