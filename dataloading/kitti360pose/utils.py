from typing import List
import numpy as np
from datapreparation.kitti360pose.imports import Object3d, Cell, Pose
from copy import deepcopy
import time
import torch
from torch_geometric.data import Data, Batch
import torch_geometric.transforms as T
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Sequence


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
            text_objects.replace("east", "east-flipped")
            .replace("west", "east")
            .replace("east-flipped", "west")
        )

        text_submap = (
            text_submap.replace("east", "east-flipped")
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
            text_objects.replace("north", "north-flipped")
            .replace("south", "north")
            .replace("north-flipped", "south")
        )

        text_submap = (
            text_submap.replace("north", "north-flipped")
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

    # for i in range(len(data_list)):
    #     data_list[i] = transform(data_list[i])

    def apply_transform(data):
        return transform(data)

    # Parallel transformation using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        # Submit all transformation tasks and store the future objects
        futures = [executor.submit(apply_transform, data) for data in data_list]

        # Wait for the futures to complete and update the data_list
        for i, future in enumerate(as_completed(futures)):
            data_list[i] = future.result()

    # time_end_transform = time.time()
    # print("time_transform: ", time_end_transform - time_start_transform)

    assert len(data_list) >= 1
    batch = Batch.from_data_list(data_list)
    return batch

def fnv_hash_vec(arr):
    '''
    FNV64-1A
    '''
    assert arr.ndim == 2
    # Floor first for negative coordinates
    arr = arr.copy()
    arr = arr.astype(np.uint64, copy=False)
    hashed_arr = np.uint64(14695981039346656037) * \
                 np.ones(arr.shape[0], dtype=np.uint64)
    for j in range(arr.shape[1]):
        hashed_arr *= np.uint64(1099511628211)
        hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
    return hashed_arr


def ravel_hash_vec(arr):
    '''
    Ravel the coordinates after subtracting the min coordinates.
    '''
    assert arr.ndim == 2
    arr = arr.copy()
    arr -= arr.min(0)
    arr = arr.astype(np.uint64, copy=False)
    arr_max = arr.max(0).astype(np.uint64) + 1

    keys = np.zeros(arr.shape[0], dtype=np.uint64)
    # Fortran style indexing
    for j in range(arr.shape[1] - 1):
        keys += arr[:, j]
        keys *= arr_max[j + 1]
    keys += arr[:, -1]
    return keys


def sparse_quantize(coords,
                    feats=None,
                    labels=None,
                    ignore_label=255,
                    set_ignore_label_when_collision=False,
                    return_index=False,
                    hash_type='fnv',
                    quantization_size=1):
    r'''Given coordinates, and features (optionally labels), the function
    generates quantized (voxelized) coordinates.

    Args:
        coords (:attr:`numpy.ndarray` or :attr:`torch.Tensor`): a matrix of size
        :math:`N \times D` where :math:`N` is the number of points in the
        :math:`D` dimensional space.

        feats (:attr:`numpy.ndarray` or :attr:`torch.Tensor`, optional): a matrix of size
        :math:`N \times D_F` where :math:`N` is the number of points and
        :math:`D_F` is the dimension of the features.

        labels (:attr:`numpy.ndarray`, optional): labels associated to eah coordinates.

        ignore_label (:attr:`int`, optional): the int value of the IGNORE LABEL.

        set_ignore_label_when_collision (:attr:`bool`, optional): use the `ignore_label`
        when at least two points fall into the same cell.

        return_index (:attr:`bool`, optional): True if you want the indices of the
        quantized coordinates. False by default.

        hash_type (:attr:`str`, optional): Hash function used for quantization. Either
        `ravel` or `fnv`. `ravel` by default.

        quantization_size (:attr:`float`, :attr:`list`, or
        :attr:`numpy.ndarray`, optional): the length of the each side of the
        hyperrectangle of of the grid cell.

    .. note::
        Please check `examples/indoor.py` for the usage.

    '''
    use_label = labels is not None
    use_feat = feats is not None
    if not use_label and not use_feat:
        return_index = True

    assert hash_type in [
        'ravel', 'fnv'
    ], "Invalid hash_type. Either ravel, or fnv allowed. You put hash_type=" + hash_type
    assert coords.ndim == 2, \
        "The coordinates must be a 2D matrix. The shape of the input is " + str(coords.shape)
    if use_feat:
        assert feats.ndim == 2
        assert coords.shape[0] == feats.shape[0]
    if use_label:
        assert coords.shape[0] == len(labels)

    # Quantize the coordinates
    dimension = coords.shape[1]
    if isinstance(quantization_size, (Sequence, np.ndarray, torch.Tensor)):
        assert len(
            quantization_size
        ) == dimension, "Quantization size and coordinates size mismatch."
        quantization_size = [i for i in quantization_size]
    elif np.isscalar(quantization_size):  # Assume that it is a scalar
        quantization_size = [quantization_size for i in range(dimension)]
    else:
        raise ValueError('Not supported type for quantization_size.')
    discrete_coords = np.floor(coords / np.array(quantization_size))

    # Hash function type
    if hash_type == 'ravel':
        key = ravel_hash_vec(discrete_coords)
    else:
        key = fnv_hash_vec(discrete_coords)

    if use_label:
        _, inds, counts = np.unique(key, return_index=True, return_counts=True)
        filtered_labels = labels[inds]
        if set_ignore_label_when_collision:
            filtered_labels[counts > 1] = ignore_label
        if return_index:
            return inds, filtered_labels
        else:
            return discrete_coords[inds], feats[inds], filtered_labels
    else:
        _, inds, inds_reverse = np.unique(key, return_index=True, return_inverse=True)
        if return_index:
            return inds, inds_reverse
        else:
            if use_feat:
                return discrete_coords[inds], feats[inds]
            else:
                return discrete_coords[inds]


import collections
import numpy as np
from scipy.linalg import expm, norm


# Rotation matrix along axis with angle theta
def M(axis, theta):
    return expm(np.cross(np.eye(3), axis / norm(axis) * theta))


class Voxelizer:

    def __init__(self,
                 voxel_size=0.006,
                 clip_bound=None,
                 use_augmentation=True,
                 scale_augmentation_bound=None,
                 rotation_augmentation_bound=None,
                 translation_augmentation_ratio_bound=None,
                 ignore_label=255):
        '''
        Args:
          voxel_size: side length of a voxel
          clip_bound: boundary of the voxelizer. Points outside the bound will be deleted
            expects either None or an array like ((-100, 100), (-100, 100), (-100, 100)).
          scale_augmentation_bound: None or (0.9, 1.1)
          rotation_augmentation_bound: None or ((np.pi / 6, np.pi / 6), None, None) for 3 axis.
            Use random order of x, y, z to prevent bias.
          translation_augmentation_bound: ((-5, 5), (0, 0), (-10, 10))
          ignore_label: label assigned for ignore (not a training label).
        '''
        self.voxel_size = voxel_size
        self.clip_bound = clip_bound
        self.ignore_label = ignore_label

        # Augmentation
        self.use_augmentation = use_augmentation
        self.scale_augmentation_bound = (0.9, 1.1)
        self.rotation_augmentation_bound = ((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi,
                                                                                          np.pi))
        self.translation_augmentation_ratio_bound = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))

    def get_transformation_matrix(self):
        voxelization_matrix, rotation_matrix = np.eye(4), np.eye(4)  # Identity
        # Get clip boundary from config or pointcloud.
        # Get inner clip bound to crop from.

        # Transform pointcloud coordinate to voxel coordinate.
        # 1. Random rotation
        rot_mat = np.eye(3)
        if self.use_augmentation and self.rotation_augmentation_bound is not None:
            if isinstance(self.rotation_augmentation_bound, collections.abc.Iterable):
                rot_mats = []
                for axis_ind, rot_bound in enumerate(self.rotation_augmentation_bound):
                    theta = 0
                    axis = np.zeros(3)
                    axis[axis_ind] = 1
                    if rot_bound is not None:
                        theta = np.random.uniform(*rot_bound)
                    rot_mats.append(M(axis, theta))
                # Use random order
                np.random.shuffle(rot_mats)
                rot_mat = rot_mats[0] @ rot_mats[1] @ rot_mats[2]
            else:
                raise ValueError()
        rotation_matrix[:3, :3] = rot_mat
        # 2. Scale and translate to the voxel space.
        scale = 1 / self.voxel_size
        if self.use_augmentation and self.scale_augmentation_bound is not None:
            scale *= np.random.uniform(*self.scale_augmentation_bound)
        np.fill_diagonal(voxelization_matrix[:3, :3], scale)  # Scale diagonal matrix, 3x3, 第四维是1，齐次矩阵方便同时进行旋转和平移
        # Get final transformation matrix.
        return voxelization_matrix, rotation_matrix

    def clip(self, coords, center=None, trans_aug_ratio=None):
        """
        Clip points outside the limit.
        return: indices of points that are inside the limit.
        """
        bound_min = np.min(coords, 0).astype(float)
        bound_max = np.max(coords, 0).astype(float)
        # print(bound_min, bound_max)
        bound_size = bound_max - bound_min
        if center is None:
            center = bound_min + bound_size * 0.5
        lim = self.clip_bound
        if trans_aug_ratio is not None:
            trans = np.multiply(trans_aug_ratio, bound_size)
            center += trans
        # Clip points outside the limit
        clip_inds = ((coords[:, 0] >= (lim[0][0] + center[0])) &
                     (coords[:, 0] < (lim[0][1] + center[0])) &
                     (coords[:, 1] >= (lim[1][0] + center[1])) &
                     (coords[:, 1] < (lim[1][1] + center[1])) &
                     (coords[:, 2] >= (lim[2][0] + center[2])) &
                     (coords[:, 2] < (lim[2][1] + center[2])))
        return clip_inds

    def voxelize(self, coords, feats, labels=None, center=None, link=None, return_ind=False):
        """
        Voxelize points.
        Using voxels instead of points, so that we can use sparse tensor. quantization
        augmentation
        """
        assert coords.shape[1] == 3 and coords.shape[0] == feats.shape[0] and coords.shape[0]
        if self.clip_bound is not None:
            trans_aug_ratio = np.zeros(3)
            if self.use_augmentation and self.translation_augmentation_ratio_bound is not None:
                for axis_ind, trans_ratio_bound in enumerate(self.translation_augmentation_ratio_bound):
                    trans_aug_ratio[axis_ind] = np.random.uniform(*trans_ratio_bound)

            # Clip points outside the limit
            clip_inds = self.clip(coords, center, trans_aug_ratio)
            if clip_inds.sum():
                coords, feats = coords[clip_inds], feats[clip_inds]
                if labels is not None:
                    labels = labels[clip_inds]

        # Get rotation and scale
        M_v, M_r = self.get_transformation_matrix()
        # Apply transformations
        rigid_transformation = M_v
        if self.use_augmentation:
            rigid_transformation = M_r @ rigid_transformation

        homo_coords = np.hstack((coords, np.ones((coords.shape[0], 1), dtype=coords.dtype)))
        coords_aug = np.floor(homo_coords @ rigid_transformation.T[:, :3])  # 当batch_size == 1: (1x4) @ (4x3)

        # Align all coordinates to the origin.
        min_coords = coords_aug.min(0)
        M_t = np.eye(4)
        M_t[:3, -1] = -min_coords
        rigid_transformation = M_t @ rigid_transformation
        coords_aug = np.floor(coords_aug - min_coords)

        # core operation
        inds, inds_reconstruct = sparse_quantize(coords_aug, return_index=True)
        coords_aug, feats, labels = coords_aug[inds], feats[inds], labels

        # Normal rotation
        if feats.shape[1] > 6:
            feats[:, 3:6] = feats[:, 3:6] @ (M_r[:3, :3].T)

        if return_ind:
            return coords_aug, feats, labels, np.array(inds_reconstruct), inds
        if link is not None:
            return coords_aug, feats, labels, np.array(inds_reconstruct), link[inds]
        return coords_aug, feats, labels, np.array(inds_reconstruct)

if __name__ == "__main__":
    def test_voxelizer():
        # 创建示例点云数据
        num_points = 100  # 点的数量
        coords = np.random.rand(num_points, 3)  # 随机生成坐标
        feats = np.random.rand(num_points, 6)  # 随机生成特征
        labels = np.random.randint(0, 10, num_points)  # 随机生成标签

        # 初始化 Voxelizer
        voxelizer = Voxelizer(
            voxel_size=0.005,
            clip_bound=((0, 1), (0, 1), (0, 1)),
            use_augmentation=True,
        )

        # 使用 Voxelizer 进行体素化
        voxelized_coords, voxelized_feats, voxelized_labels, inds_reconstruct, index = voxelizer.voxelize(coords, feats,
                                                                                                   labels,
                                                                                                   return_ind=True)


        # 打印结果
        print("原始坐标:", coords.shape)
        print("体素化坐标:", voxelized_coords.shape)
        print("原始特征:", feats.shape)
        print("体素化特征:", voxelized_feats.shape)
        print("原始标签:", labels.shape)
        print("体素化标签:", voxelized_labels.shape)
        print("重建索引:", inds_reconstruct.shape)
        print(inds_reconstruct)

    # 运行测试
    test_voxelizer()