import numpy as np

processed_mask3d_data_path = "/home/wanglichao/Mask3D/data/processed/t2p/train/scene_0_cell_0.npy"
np_f1 = np.load(processed_mask3d_data_path)
print(np_f1.shape)
print(np_f1[0])