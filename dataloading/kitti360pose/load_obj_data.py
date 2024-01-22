import time
from typing import List

import os
import os.path as osp
import pickle as pkl
import numpy as np
import cv2
from copy import deepcopy
from dataloading.kitti360pose.cells import *

# with open("/home/wanglichao/Text2Position/data/k360_text_30-10_gridCells_pd10_pc4_shiftPoses_all_nm-6/cells/2013_05_28_drive_0000_sync.pkl", "rb") as f:
#     cells = pkl.load(f)
#     print(len(cells))
#     print(cells[69], cells[69].get_center())
#     for obj in cells[69].objects:
#         print(obj, obj.id, obj.instance_id)
#     # print(cells[0])
#     # print(cells[0].objects[0])
#     # print(cells[0].objects[0].xyz.shape)
#     # print(cells[0].objects[0].feature_2d)
# #
# with open(
#         "/home/wanglichao/Text2Position/data/k360_30-10_scG_pd10_pc4_spY_all/cells/2013_05_28_drive_0000_sync.pkl",
#         "rb") as f:
#     cells = pkl.load(f)
#     print(len(cells))
#     print(cells[69], cells[69].get_center())
#     for obj in cells[69].objects:
#         print(obj, obj.id, obj.instance_id)
#     # print(cells[0].objects[0])
#     # print(cells[0].objects[0].xyz.shape)

# with open(
#         "/home/wanglichao/Text2Position/data/k360_30-10_scG_pd10_pc4_spY_all/cells/2013_05_28_drive_0000_sync.pkl",
#         "rb") as f:
#     cells = pkl.load(f)
#     for cell in cells:
#         for obj in cell.objects:
#             if obj.instance_id == 17144:
#                 print(obj, obj.id, obj.instance_id)

# with open("/home/wanglichao/Text2Position/data/k360_30-10_scG_pd10_pc4_spY_all/poses/2013_05_28_drive_0000_sync.pkl", "rb") as f:
#     poses = pkl.load(f)
#     print(len(poses))
#     print(poses[0])
#     loc = poses[0].pose_w[0:2]
#     for descr in poses[0].descriptions:
#         print(descr)
#         print(descr.offset_closest)

# with open("/home/wanglichao/KITTI-360/objects/2013_05_28_drive_0003_sync.pkl", "rb") as f:
#     objs = pkl.load(f)
#     print(len(objs))
#     n_2d_feature = 0
#     for obj in objs:
#         print(obj, obj.instance_id, obj.label, obj.get_center(), obj.xyz.shape)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle as pkl
import clip


with open("/home/wanglichao/KITTI-360/objects/2013_05_28_drive_0003_sync.pkl", "rb") as f:
    objs = pkl.load(f)

# 创建3D图形
all_xyz = np.concatenate([obj.xyz for obj in objs], axis=0)
global_min_xyz = np.min(all_xyz, axis=0)
global_max_xyz = np.max(all_xyz, axis=0)
print(global_min_xyz, global_max_xyz)
# 遍历每个对象并绘制它的点云

for obj in objs:

    xyz = obj.xyz  # 假设这是点云坐标的数组
    rgb = obj.rgb  # 假设这是点云颜色的数组
    print(len(obj.image_2d))
    if obj.feature_2d is None:
        print("None", obj.instance_id)
        continue
    else:
        print(obj.feature_2d.shape)

    if obj.instance_id == 7000:
        image2d = obj.image_2d

        # print(len(image2d))
        # print(image2d[0][0].shape)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # # print(obj.instance_id, obj.label, xyz.shape)
    # # ax.scatter(normalized_xyz[:, 0], normalized_xyz[:, 1], normalized_xyz[:, 2], s=1)  # 绘制正则化的点云
    # ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2])  # 绘制正则化的点云
    #
    # # 设置轴标签
    # ax.set_xlabel('X Axis')
    # ax.set_ylabel('Y Axis')
    # ax.set_zlabel('Z Axis')
    # ax.set_title(f"{obj.instance_id}, {obj.label}")
    #
    # ax.set_xlim([global_min_xyz[0], global_max_xyz[0]])
    # ax.set_ylim([global_min_xyz[1], global_max_xyz[1]])
    # ax.set_zlim([global_min_xyz[2], global_max_xyz[2]])
    #
    # # 显示图形
    # plt.show()
    # plt.close()



# base_path = "/home/wanglichao/Text2Pos-CVPR2022/data/k360_30-10_scG_pd10_pc4_spY_all/"
# batch_size = 64
# transform = T.FixedPoints(256)
#
# base_path_new = "/home/wanglichao/Text2Pos-CVPR2022/data/k360_30-10_scG_pd10_pc4_spY_all_new/"
# os.mkdir(base_path_new) if not os.path.exists(base_path_new) else None
# os.mkdir(base_path_new + "cells") if not os.path.exists(base_path_new + "cells") else None
# os.mkdir(base_path_new + "poses") if not os.path.exists(base_path_new + "poses") else None
#
# # Cells preparation
# for scene_names in (SCENE_NAMES, SCENE_NAMES_TRAIN, SCENE_NAMES_VAL, SCENE_NAMES_TEST):
#     dataset_multi = Kitti360CoarseDatasetMulti(
#         base_path, scene_names, transform, shuffle_hints=False, flip_poses=False
#     )
#     dataloader_multi = DataLoader(
#         dataset_multi,
#         batch_size=batch_size,
#         collate_fn=Kitti360CoarseDataset.collate_fn,
#         shuffle=False,
#     )
#     print(time.time())
#     time_start = time.time()
#     for i_batch, batch in enumerate(dataloader_multi):
#         print(f"idx {i_batch}, scene {batch['scene_names']}, cell {batch['cell_ids']}")
#     print(time.time())
#     time_end = time.time()
#     print(time_end - time_start)

