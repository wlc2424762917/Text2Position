import time
from typing import List
import sys
sys.path.append("/home/wanglichao/Text2Position")
import os
import os.path as osp
import pickle as pkl
import numpy as np
import cv2
from copy import deepcopy
from dataloading.kitti360pose.cells import *
import clip


scene_base_path = "/home/wanglichao/Text2Position/data/k360_30-10_scG_pd10_pc4_spY_all/"
# scene_base_path = "/home/wanglichao/Text2Position/data/k360_text_30-10_gridCells_pd10_pc4_shiftPoses_all_nm-6/"
object_base_path = "/home/wanglichao/KITTI-360/objects/"

cell_files = os.listdir(scene_base_path + "cells")
cell_files.sort()
print(f"Total cells: {cell_files}.")

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

for scene_name in cell_files:
    cell_path = scene_base_path + "cells/" + scene_name
    objs_path = object_base_path + scene_name

    with open(cell_path, "rb") as f, open(objs_path, "rb") as f2:
        print(f"Loading {cell_path} and {objs_path}.")
        cells = pkl.load(f)
        objs = pkl.load(f2)
        objs_dict = {obj.instance_id: obj for obj in objs}

        for cell in cells:
            for cell_obj in cell.objects:
                match = objs_dict.get(cell_obj.instance_id)
                if match:
                    with torch.no_grad():
                        text = clip.tokenize(cell_obj.label).to(device)
                        text_features = model.encode_text(text)
                        text_features /= text_features.norm(dim=-1, keepdim=True)
                        text_features = text_features.cpu().numpy()
                    if match.feature_2d is None:
                        cell_obj.feature_2d = text_features
                    else:
                        cell_obj.feature_2d = match.feature_2d * 0.4 + 0.6 * text_features
                    print(f"success: {cell_obj.instance_id}, {cell_obj.label}")
                else:
                    with torch.no_grad():
                        text = clip.tokenize(cell_obj.label).to(device)
                        text_features = model.encode_text(text)
                        text_features /= text_features.norm(dim=-1, keepdim=True)
                        cell_obj.feature_2d = text_features.cpu().numpy()
                    print(f"not found, using clip: {cell_obj.instance_id}, {cell_obj.label}")
                    # print(f"error: no matching object found for {cell.id}, {cell_obj.instance_id}, {cell_obj.label}, {cell_obj.get_color_rgb()}")

    save_path = f"/home/wanglichao/Text2Position/data/k360_30-10_scG_pd10_pc4_spY_all_clip_feature/cells/{scene_name}"
    # 创建保存目录（如果不存在）
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pkl.dump(cells, f)
        print(f"Data saved to {save_path}")





