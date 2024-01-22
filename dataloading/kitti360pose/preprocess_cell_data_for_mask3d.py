import glob
import time
from typing import List

import os
import os.path as osp
import pickle as pkl
import numpy as np
import cv2
from copy import deepcopy
from dataloading.kitti360pose.cells import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle as pkl
import clip
import pandas as pd
import numpy as np
import pickle as pkl

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
base_dir = "/home/wanglichao/Text2Position/data/k360_30-10_scG_pd10_pc4_spY_all/cells_txt"
scenes = glob.glob("/home/wanglichao/Text2Position/data/k360_30-10_scG_pd10_pc4_spY_all/cells/*")
scenes = sorted(scenes)
print(scenes)
# 加载数据
for scene_idx, scene in enumerate(scenes):
    print(scene)
    # 为每个scene创建一个子目录
    scene_dir = os.path.join(base_dir, f"scene_{scene_idx}")
    if not os.path.exists(scene_dir):
        os.makedirs(scene_dir)
    else:
        print(f"Scene {scene_idx} already exists!")
        continue

    with open(scene, "rb") as f:
        cells = pkl.load(f)
        for cell_idx, cell in enumerate(cells):
            objs = cell.objects

            points = []
            for obj in objs:
                xyz, rgb, sem_label, inst_label = obj.xyz, obj.rgb, obj.label, obj.id
                sem_label = CLASS_TO_INDEX[sem_label]
                for i in range(len(xyz)):
                    # Ensure all components are np.float64
                    point = np.concatenate((xyz[i].astype(np.float64),
                                            rgb[i].astype(np.float64),
                                            np.array([sem_label], dtype=np.float64),
                                            np.array([inst_label], dtype=np.float64)))
                    points.append(point)

            # Convert points to a DataFrame
            df_points = pd.DataFrame(points, columns=['X', 'Y', 'Z', 'R', 'G', 'B', 'Sem_Label', 'Inst_Label'])

            # Save as CSV file
            csv_filename = os.path.join(scene_dir, f"scene_{scene_idx}_cell_{cell_idx}.csv")
            df_points.to_csv(csv_filename, index=False, header=False)

            # Print the shape of points
            # print(df_points.shape)