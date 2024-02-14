import pandas as pd

import os
import shutil


def move_files_to_parent_directory(folder_path):
    # 获取父目录的路径
    parent_dir = os.path.dirname(folder_path)

    # 遍历指定文件夹中的所有文件和文件夹
    for filename in os.listdir(folder_path):
        # 获取文件或文件夹的完整路径
        file_path = os.path.join(folder_path, filename)

        # 检查是否为文件（不是文件夹）
        if os.path.isfile(file_path):
            # 构建目标路径（父目录中的路径）
            destination = os.path.join(parent_dir, filename)

            # 移动文件
            shutil.move(file_path, destination)
            print(f"Moved: {filename}")


# 使用示例
# parent_folder = '/home/wanglichao/Text2Position/data/k360_30-10_scG_pd10_pc4_spY_all/cells_txt/validation/'
# parent_folder = '/home/wanglichao/Text2Position/data/k360_30-10_scG_pd10_pc4_spY_all/cells_txt/test/'
parent_folder = '/home/wanglichao/Text2Position/data/k360_30-10_scG_pd10_pc4_spY_all/cells_txt/train/'

folder_paths = os.listdir(parent_folder)
# folder_path = '/home/wanglichao/Text2Position/data/k360_30-10_scG_pd10_pc4_spY_all/cells_csv/validation/scene_2'  # 替换为你的文件夹路径
for folder_path in folder_paths:
    folder_path = os.path.join(parent_folder, folder_path)
    move_files_to_parent_directory(folder_path)


# import numpy as np
# file1 = "/home/wanglichao/Text2Position/data/k360_30-10_scG_pd10_pc4_spY_all/cells_txt/train/scene_0_cell_0.csv"
# file1 = pd.read_csv(file1, header=None).values
#
#
# file1 = file1.astype(np.float64)
# for item in file1[1]:
#     print(item, type(item))
#
# file2 = "/home/wanglichao/Synthetic_v3_InstanceSegmentation/train/1_points_GTv3.txt"
# file2 = pd.read_csv(file2, header=None).values
# for item in file2[1]:
#     print(item, type(item))