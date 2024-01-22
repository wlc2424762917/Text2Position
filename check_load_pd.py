import pandas as pd

import os
import shutil




import numpy as np
file1 = "/home/wanglichao/Mask3D/data/processed/t2p/test/scene_0_cell_0.csv.npy"
np_f1 = np.load(file1)
print(np_f1.shape)
for item in np_f1[1]:
    print(item, type(item))

# file2 = "/home/wanglichao/Synthetic_v3_InstanceSegmentation/train/1_points_GTv3.txt"
# file2 = pd.read_csv(file2, header=None).values
# for item in file2[1]:
#     print(item, type(item))