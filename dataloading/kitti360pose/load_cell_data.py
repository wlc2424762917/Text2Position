import time
from typing import List

import os
import os.path as osp
import pickle as pkl
import numpy as np
import cv2
from copy import deepcopy
from dataloading.kitti360pose.cells import *

with open("/home/wanglichao/Text2Position/data/k360_text_30-10_gridCells_pd10_pc4_shiftPoses_all_nm-6/cells/2013_05_28_drive_0002_sync.pkl", "rb") as f:
    cells = pkl.load(f)
    print(len(cells))
    print(cells[0])
    print(cells[0].objects[0])
    print(cells[0].objects[0].xyz.shape)
    print(cells[0].objects[0].feature_2d)

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
#         print(obj)
#         if obj.feature_2d is not None:
#             n_2d_feature += 1
#             print(obj.feature_2d.shape)
#         else:
#             print("None 2D feature")
#     print(f"n_2d_feature: {n_2d_feature} out of {len(objs)}")

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

