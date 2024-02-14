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
from datapreparation.kitti360pose.utils import randomize_class_and_color
import MinkowskiEngine as ME

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

        self.hint_descriptions_alt = [
            Kitti360BaseDataset.create_hint_description_alt(pose, self.cells_dict[pose.cell_id])
            for pose in self.poses
        ]
        self.submap_descriptions_alt = [
            Kitti360BaseDataset.create_submap_description_alt(self, pose, self.cells_dict[pose.cell_id])
            for pose in self.poses
        ]

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
            # descr.direction, descr.object_color_text, descr.object_label = ov_change_descr(descr.direction, descr.object_color_text, descr.object_label)
            hints.append(
                f"The pose is {descr.direction} of a {descr.object_color_text} {descr.object_label}."
            )
        # time_end_create_hint_description = time.time()
        # print("time_create_hint_description: ", time_end_create_hint_description - time_start_create_hint_description)
        return hints

    def create_hint_description_alt(pose: Pose, cell: Cell):
        hints = []
        # cell_objects_dict = {obj.id: obj for obj in cell.objects}
        # time_start_create_hint_description = time.time()
        for descr in pose.descriptions:
            # obj = cell_objects_dict[descr.object_id]
            # hints.append(f'The pose is {descr.direction} of a {obj.get_color_text()} {obj.label}.')
            object_color_text, object_label = randomize_class_and_color(descr.object_label, descr.object_color_text)
            hints.append(
                f"The pose is {descr.direction} of a {object_color_text} {object_label}."
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

    def create_submap_description_alt(self, pose: Pose, cell: Cell):
        # print("create_submap_description_alt")
        hints = []
        # cell_objects_dict = {obj.id: obj for obj in cell.objects}
        for descr in pose.descriptions:
            # obj = cell_objects_dict[descr.object_id]
            # hints.append(f'The pose is {descr.direction} of a {obj.get_color_text()} {obj.label}.')
            new_direction = self.direction_map[descr.direction]
            object_color_text, object_label = randomize_class_and_color(descr.object_label, descr.object_color_text)
            hints.append(
                f"The submap has a {object_color_text} {object_label} on {new_direction}."
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
        # print(batch.keys())
        # print(batch["cells_features"])
        # quit()
        return batch

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

if __name__ == "__main__":
    base_path = "./data/k360_decouple"
    folder_name = "2013_05_28_drive_0003_sync"

    dataset = Kitti360BaseDataset(base_path, folder_name)
