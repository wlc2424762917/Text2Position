import time
from typing import List

import os
import os.path as osp
import pickle as pkl
import numpy as np
import cv2
from copy import deepcopy
from dataloading.kitti360pose.cells import *
import clip

model, preprocess = clip.load("ViT-B/32", device='cuda')
with open("/home/wanglichao/KITTI-360/objects/2013_05_28_drive_0003_sync.pkl", "rb") as f:
    objs = pkl.load(f)
    print(len(objs))
    n_2d_feature = 0
    for obj in objs:
        print(obj)
        if obj.feature_2d is not None:
            n_2d_feature += 1
            print(obj.feature_2d.shape, obj.feature_2d.max(), obj.feature_2d.min())
        else:
            print("None 2D feature")
            text = clip.tokenize(obj.label).to('cuda')
            text_features = model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            obj.feature_2d = text_features.cpu().numpy()
            print(obj.feature_2d.shape, obj.feature_2d.max(), obj.feature_2d.min())
    print(f"n_2d_feature: {n_2d_feature} out of {len(objs)}")

