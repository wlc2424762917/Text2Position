# from typing import List
#
# import os
# import os.path as osp
# import pickle as pkl
# import numpy as np
# import cv2
# from copy import deepcopy
#
# import torch
# from torch.utils.data import Dataset, DataLoader
#
# import torch_geometric.transforms as T
#
# from datapreparation.kitti360pose.utils import (
#     CLASS_TO_LABEL,
#     LABEL_TO_CLASS,
#     CLASS_TO_MINPOINTS,
#     SCENE_NAMES,
#     SCENE_NAMES_TEST,
#     SCENE_NAMES_TRAIN,
#     SCENE_NAMES_VAL,
# )
# from datapreparation.kitti360pose.utils import CLASS_TO_INDEX, COLOR_NAMES
# from datapreparation.kitti360pose.imports import Object3d, Cell, Pose
# from datapreparation.kitti360pose.drawing import (
#     show_pptk,
#     show_objects,
#     plot_cell,
#     plot_pose_in_best_cell,
# )
# from dataloading.kitti360pose.base import Kitti360BaseDataset
# from dataloading.kitti360pose.utils import batch_object_points, flip_pose_in_cell
#
# with open("/home/wanglichao/Text2Pos-CVPR2022/data/k360_30-10_scG_pd10_pc4_spY_all/cells/2013_05_28_drive_0000_sync.pkl", "rb") as f:
#     cells = pkl.load(f)
#     print(len(cells))
#     print(type(cells[0]))
# #
# # with open("/home/wanglichao/Text2Pos-CVPR2022/data/k360_30-10_scG_pd10_pc4_spY_all/poses/2013_05_28_drive_0000_sync.pkl", "rb") as f:
# #     objects = pkl.load(f)
# #     print(len(objects))
direction_map = {"top": "base", "north": "south", "south": "north", "east": "west", "west": "east"}
print(direction_map["top"])