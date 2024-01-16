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
from dataloading.kitti360pose.cells import *
#
# def save_point_cloud_to_ply(xyz, rgb, filename):
#     assert xyz.shape[0] == rgb.shape[0], "点的数量必须与颜色的数量相等"
#
#     # 检查颜色值范围并将其转换为 [0, 255]
#     if rgb.max() <= 1:
#         rgb = (rgb * 255).astype(np.uint8)
#     else:
#         rgb = rgb.astype(np.uint8)
#
#     # 将点云数据组织为 PLY 格式
#     points = np.hstack([xyz, rgb])
#     header = '''ply
# format ascii 1.0
# element vertex {}
# property float32 x
# property float32 y
# property float32 z
# property uchar red
# property uchar green
# property uchar blue
# end_header
# '''.format(len(points))
#
#     # 写入文件
#     with open(filename, 'w') as ply_file:
#         ply_file.write(header)
#         for point in points:
#             ply_file.write('{} {} {} {} {} {}\n'.format(*point))
#
# # 加载数据
# with open("/home/wanglichao/Text2Position/data/k360_30-10_scG_pd10_pc4_spY_all/cells/2013_05_28_drive_0000_sync.pkl", "rb") as f:
#     cells = pkl.load(f)
#     cell = cells[100]
#     objs = cell.objects
#
# # 创建点云数据
# all_xyz = np.concatenate([obj.xyz for obj in objs], axis=0)
# all_rgb = np.concatenate([obj.rgb for obj in objs], axis=0)
#
# # 保存点云数据为 PLY 文件
# save_point_cloud_to_ply(all_xyz, all_rgb, '/home/wanglichao/cell_point_cloud.ply')
#
#

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
import numpy as np
import pickle as pkl
import open3d as o3d

# 加载数据
with open("/home/wanglichao/Text2Position/data/k360_30-10_scG_pd10_pc4_spY_all/cells/2013_05_28_drive_0000_sync.pkl", "rb") as f:
    cells = pkl.load(f)
    cell = cells[100]
    objs = cell.objects

# 创建点云数据
all_xyz = np.concatenate([obj.xyz for obj in objs], axis=0)
all_rgb = np.concatenate([obj.rgb for obj in objs], axis=0)
all_labels = np.array([CLASS_TO_INDEX[obj.label] for obj in objs])
all_instance_ids = np.array([obj.id for obj in objs])

# 创建 Open3D 点云对象
pcd = o3d.geometry.PointCloud()

# 设置点云的坐标
pcd.points = o3d.utility.Vector3dVector(all_xyz)

# 检查颜色值范围并转换为 [0, 1] 范围内的浮点数
if all_rgb.max() > 1:
    all_rgb = all_rgb / 255.0

# 设置点云的颜色
pcd.colors = o3d.utility.Vector3dVector(all_rgb)

# 保存点云为 PLY 文件
o3d.io.write_point_cloud("/home/wanglichao/cell_point_cloud_open3d.ply", pcd)
