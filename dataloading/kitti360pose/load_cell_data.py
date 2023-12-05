import time
from typing import List

import os
import os.path as osp
import pickle as pkl
import numpy as np
import cv2
from copy import deepcopy
from dataloading.kitti360pose.cells import *

# with open("/home/wanglichao/Text2Pos-CVPR2022/data/k360_30-10_scG_pd10_pc4_spY_all/cells/2013_05_28_drive_0000_sync.pkl", "rb") as f:
#     cells = pkl.load(f)
#     print(len(cells))
#     print(cells[0])
#     print(cells[0].objects[0])
#     print(cells[0].objects[0].xyz.shape)

base_path = "/home/wanglichao/Text2Pos-CVPR2022/data/k360_30-10_scG_pd10_pc4_spY_all/"
batch_size = 64
transform = T.FixedPoints(256)

base_path_new = "/home/wanglichao/Text2Pos-CVPR2022/data/k360_30-10_scG_pd10_pc4_spY_all_new/"
os.mkdir(base_path_new) if not os.path.exists(base_path_new) else None
os.mkdir(base_path_new + "cells") if not os.path.exists(base_path_new + "cells") else None
os.mkdir(base_path_new + "poses") if not os.path.exists(base_path_new + "poses") else None

# Cells preparation
for scene_names in (SCENE_NAMES, SCENE_NAMES_TRAIN, SCENE_NAMES_VAL, SCENE_NAMES_TEST):
    dataset_multi = Kitti360CoarseDatasetMulti(
        base_path, scene_names, transform, shuffle_hints=False, flip_poses=False
    )
    dataloader_multi = DataLoader(
        dataset_multi,
        batch_size=batch_size,
        collate_fn=Kitti360CoarseDataset.collate_fn,
        shuffle=False,
    )
    print(time.time())
    time_start = time.time()
    for i_batch, batch in enumerate(dataloader_multi):
        print(f"idx {i_batch}, scene {batch['scene_names']}, cell {batch['cell_ids']}")
    print(time.time())
    time_end = time.time()
    print(time_end - time_start)

