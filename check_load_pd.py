import pandas as pd

import os
import shutil




import numpy as np
# file1 = "/home/wanglichao/Mask3D/data/processed/t2p/validation/scene_8_cell_87.npy"
# np_f1 = np.load(file1)
# print(np_f1.shape)
# for item in np_f1[2000]:
#     print(item, type(item))
#
file1 = "/home/wanglichao/Mask3D/data/processed/t2p/validation/scene_8_cell_87_0.npy"
np_f1 = np.load(file1)
print(np_f1.shape)
for item in np_f1[0]:
    print(item, type(item))

# #
# file1 = "/home/wanglichao/Mask3D/data/processed/t2p/train/scene_0_cell_0.npy"
# np_f1 = np.load(file1)
# print(np_f1.shape)
# for item in np_f1[500]:
#     print(item, type(item))
# #
# file1 = "/home/wanglichao/Mask3D/data/processed/stpls3d/test/11_points_GTv3_0.npy"
# np_f1 = np.load(file1)
# print(np_f1.shape)
# for item in np_f1[500]:
#     print(item, type(item))

# file1 = "/home/wanglichao/Mask3D/data/processed/stpls3d/validation/6_points_GTv3_1.npy"
# np_f1 = np.load(file1)
# print(np_f1.shape)
# for item in np_f1[500]:
#     print(item, type(item))
# print("\n")
file2 = "/home/wanglichao/Mask3D/data/processed/t2p/instance_gt/validation/scene_8_cell_87_0.txt"
file2 = pd.read_csv(file2, header=None).values
print(file2.shape)
for line in file2:
    print(line)
# for line in file2:
#     if line[-2] != 0:
#         print(line)
#         print("\n")

# file2 = "/home/wanglichao/Text2Position/data/k360_30-10_scG_pd10_pc4_spY_all/cells_txt/validation/scene_8_cell_87.csv"
# file2 = pd.read_csv(file2, header=None).values
# points = file2
# print(points[0])
#
# points[:, 0:3] = points[:, 0:3] * 30
# points[:, 3:6] = points[:, 3:6] * 255
#
# for item in points:
#     print(item[0:3])