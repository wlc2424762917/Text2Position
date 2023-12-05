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
#

base_path = "/home/wanglichao/Text2Pos-CVPR2022/data/k360_30-10_scG_pd10_pc4_spY_all/"
batch_size = 1
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
    for i_batch, batch in enumerate(dataloader_multi):
        print(f"idx {i_batch}, scene {batch['scene_names']}, cell {batch['cell_ids']}")
        os.mkdir(base_path_new + "cells/" + batch["scene_names"][0] + batch["cell_ids"][0]) if not os.path.exists(
            base_path_new + "cells/" + batch["scene_names"][0] + batch["cell_ids"][0]) else None

        # save text
        with open(base_path_new + "cells/" + batch["scene_names"][0] + batch["cell_ids"][0] + "/text.txt", "w") as f:
            f.write(batch["texts"][0])

        # save objects
        with open(base_path_new + "cells/" + batch["scene_names"][0] + batch["cell_ids"][0] + "/objects.pkl", "wb") as f:
            pkl.dump(batch["objects"][0], f)

        # save object_points
        with open(base_path_new + "cells/" + batch["scene_names"][0] + batch["cell_ids"][0] + "/object_points.pkl", "wb") as f:
            pkl.dump(batch["object_points"][0], f)

        # save cell
        with open(base_path_new + "cells/" + batch["scene_names"][0] + batch["cell_ids"][0] + "/cell.pkl", "wb") as f:
            pkl.dump(batch["cells"][0], f)

        # save pose
        with open(base_path_new + "poses/" + batch["scene_names"][0] + batch["cell_ids"][0] + ".pkl", "wb") as f:
            pkl.dump(batch["poses"][0], f)






        # "poses": pose,
        # "cells": cell,
        # "objects": cell.objects,
        # "object_points": object_points,
        # "texts": text,
        # "texts_objects": text_objects,
        # "texts_submap": text_submap,
        # "cell_ids": pose.cell_id,
        # "scene_names": self.scene_name,
        # "object_class_indices": object_class_indices,
        # "object_color_indices": object_color_indices,
        # "debug_hint_descriptions": hints,  # Care: Not shuffled etc!
