import numpy as np
import pickle as pkl
import time
from typing import List

import os
import os.path as osp
import pickle as pkl
import numpy as np
import cv2
from copy import deepcopy
import open3d as o3d

from dataloading.kitti360pose.cells import *
CLASS_TO_INDEX = {
    "building": 0,
    "pole": 1,
    "traffic light": 2,
    "traffic sign": 3,
    "garage": 4,
    "stop": 5,
    "smallpole": 6,
    "lamp": 7,
    "trash bin": 8,
    "vending machine": 9,
    "box": 10,
    "road": 11,
    "sidewalk": 12,
    "parking": 13,
    "wall": 14,
    "fence": 15,
    "guard rail": 16,
    "bridge": 17,
    "tunnel": 18,
    "vegetation": 19,
    "terrain": 20,
    "pad": 21,
}

def assign_colors_to_labels(labels):
    unique_labels = np.unique(labels)
    colors = np.random.rand(len(unique_labels), 3)  # 为每个唯一标签生成随机颜色
    label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
    return np.array([label_to_color[label] for label in labels])


with open("/home/wanglichao/Text2Position/data/k360_30-10_scG_pd10_pc4_spY_all/cells/2013_05_28_drive_0000_sync.pkl", "rb") as f:
    cells = pkl.load(f)
    cell = cells[100]
    objs = cell.objects
    for obj in objs:
        print(obj, obj.instance_id, obj.label)

# 创建点云数据
all_xyz = np.concatenate([obj.xyz for obj in objs], axis=0)
all_labels = np.concatenate([np.full(len(obj.xyz), CLASS_TO_INDEX[obj.label]) for obj in objs])

# 创建 Open3D 点云对象
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(all_xyz)

# 为每个标签赋予颜色
label_colors = assign_colors_to_labels(all_labels)
pcd.colors = o3d.utility.Vector3dVector(label_colors)

# 保存点云为 PLY 文件
o3d.io.write_point_cloud("/home/wanglichao/cell_point_cloud_labels_open3d.ply", pcd)
