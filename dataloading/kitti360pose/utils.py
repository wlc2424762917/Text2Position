from typing import List
import numpy as np
from datapreparation.kitti360pose.imports import Object3d, Cell, Pose
from copy import deepcopy
import time
import torch
from torch_geometric.data import Data, Batch
import torch_geometric.transforms as T
from concurrent.futures import ThreadPoolExecutor, as_completed


# TODO: for free orientations, possibly flip cell only, create descriptions and hints again
# OR: numeric vectors in descriptions, flip cell objects and description.direction, then create hints again
# Flip pose, too?
def flip_pose_in_cell(pose: Pose, cell: Cell, text, text_objects, text_submap, direction, hints=None, offsets=None):
    """Flips the cell horizontally or vertically
    CARE: Needs adjustment for non-compass directions
    CARE: Description.object_closest_point is flipped but direction in description is not flipped.

    Args:
        pose (Pose): The pose to flip, is copied before modification
        cell (Cell): The cell to flip, is copied before modification
        text (str): The text description to flip
        direction (int): Horizontally (+1) or vertically (-1)

    Returns:
        Pose: flipped pose
        Cell: flipped cell
        str: flipped text
    """
    assert direction in (-1, 1)
    assert sum([hints is None, offsets is None]) != 1  # Either both or none

    pose = deepcopy(pose)
    cell = deepcopy(cell)
    if offsets is not None:
        offsets = offsets.copy()

    if direction == 1:  # Horizontally
        pose.pose[0] = 1.0 - pose.pose[0]
        for obj in cell.objects:
            obj.xyz[:, 0] = 1 - obj.xyz[:, 0]
        for descr in pose.descriptions:
            descr.closest_point[0] = 1.0 - descr.closest_point[0]

        text = (
            text.replace("east", "east-flipped")
            .replace("west", "east")
            .replace("east-flipped", "west")
        )

        text_objects = (
            text.replace("east", "east-flipped")
            .replace("west", "east")
            .replace("east-flipped", "west")
        )

        text_submap = (
            text.replace("east", "east-flipped")
            .replace("west", "east")
            .replace("east-flipped", "west")
        )

        if hints is not None:
            hints = [
                hint.replace("east", "east-flipped")
                .replace("west", "east")
                .replace("east-flipped", "west")
                for hint in hints
            ]
            offsets[:, 0] *= -1

    elif direction == -1:  # Vertically
        pose.pose[1] = 1.0 - pose.pose[1]
        for obj in cell.objects:
            obj.xyz[:, 1] = 1 - obj.xyz[:, 1]
        for descr in pose.descriptions:
            descr.closest_point[1] = 1.0 - descr.closest_point[1]

        text = (
            text.replace("north", "north-flipped")
            .replace("south", "north")
            .replace("north-flipped", "south")
        )

        text_objects = (
            text.replace("north", "north-flipped")
            .replace("south", "north")
            .replace("north-flipped", "south")
        )

        text_submap = (
            text.replace("north", "north-flipped")
            .replace("south", "north")
            .replace("north-flipped", "south")
        )

        if hints is not None:
            hints = [
                hint.replace("north", "north-flipped")
                .replace("south", "north")
                .replace("north-flipped", "south")
                for hint in hints
            ]
            offsets[:, 1] *= -1

    assert "flipped" not in text

    if hints is not None:
        return pose, cell, text, text_objects, text_submap, hints, offsets
    else:
        return pose, cell, text, text_objects, text_submap

def normalize_scale_vectorized(xyz_array):
    # Centering each point cloud
    centers = np.mean(xyz_array, axis=1, keepdims=True)
    xyz_centered = xyz_array - centers

    # Normalizing to (-1, 1)
    max_scales = np.max(np.abs(xyz_centered), axis=(1, 2), keepdims=True)
    xyz_normalized = xyz_centered / max_scales * 0.999999
    return xyz_normalized


def random_rotate_vectorized(xyz_array, axis=2):
    batch_size = xyz_array.shape[0]
    degrees = np.random.uniform(-120, 120, size=batch_size)
    radians = np.radians(degrees)

    # Creating rotation matrices for the entire batch
    sin_vals, cos_vals = np.sin(radians), np.cos(radians)
    if axis == 2:  # z-axis rotation
        rotation_matrices = np.array([
            [cos_vals, -sin_vals, np.zeros(batch_size)],
            [sin_vals, cos_vals, np.zeros(batch_size)],
            [np.zeros(batch_size), np.zeros(batch_size), np.ones(batch_size)]
        ]).transpose(1, 2, 0)

    # Apply rotation to each point cloud
    return np.einsum('bij,bjk->bik', xyz_array, rotation_matrices)

def batch_object_points(objects: List[Object3d], transform):
    """Generates a PyG-Batch for the objects of a single cell.
    Note: Aggregating an entire batch of cells into a single PyG-Batch would exceed the limit of 256 sub-graphs.
    Note: The objects can be transformed / augmented freely, as their center-points are encoded separately.

    Args:
        objects (List[Object3d]): Cell objects
        transform: PyG-Transform
    """

    # CARE: Transforms not working with batches?! Doing it object-by-object here!
    data_list = [
        Data(
            x=torch.tensor(obj.rgb, dtype=torch.float), pos=torch.tensor(obj.xyz, dtype=torch.float)
        )
        for obj in objects
    ]

    # time_start_transform = time.time()

    for i in range(len(data_list)):
        data_list[i] = transform(data_list[i])

    # def apply_transform(data):
    #     return transform(data)
    #
    # # Parallel transformation using ThreadPoolExecutor
    # with ThreadPoolExecutor() as executor:
    #     # Submit all transformation tasks and store the future objects
    #     futures = [executor.submit(apply_transform, data) for data in data_list]
    #
    #     # Wait for the futures to complete and update the data_list
    #     for i, future in enumerate(as_completed(futures)):
    #         data_list[i] = future.result()

    # time_end_transform = time.time()
    # print("time_transform: ", time_end_transform - time_start_transform)

    assert len(data_list) >= 1
    batch = Batch.from_data_list(data_list)
    return batch
