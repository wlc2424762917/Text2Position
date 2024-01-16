import numpy as np
import pickle as pkl
import open3d as o3d
from dataloading.kitti360pose.cells import *


def assign_colors_to_instances(instance_ids):
    unique_ids = np.unique(instance_ids)
    colors = np.random.rand(len(unique_ids), 3)  # 为每个唯一实例 ID 生成随机颜色
    id_to_color = {uid: colors[i] for i, uid in enumerate(unique_ids)}
    return np.array([id_to_color[uid] for uid in instance_ids])

# 加载数据
with open("/home/wanglichao/Text2Position/data/k360_30-10_scG_pd10_pc4_spY_all/cells/2013_05_28_drive_0000_sync.pkl", "rb") as f:
    cells = pkl.load(f)
    cell = cells[100]
    objs = cell.objects
    for obj in objs:
        print(obj, obj.instance_id, obj.label)

# 创建点云数据
all_xyz = np.concatenate([obj.xyz for obj in objs], axis=0)
all_instance_ids = np.concatenate([np.full(len(obj.xyz), obj.id) for obj in objs])

# 创建 Open3D 点云对象
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(all_xyz)

# 为每个实例赋予颜色
instance_colors = assign_colors_to_instances(all_instance_ids)
pcd.colors = o3d.utility.Vector3dVector(instance_colors)

# 保存点云为 PLY 文件
o3d.io.write_point_cloud("/home/wanglichao/cell_point_cloud_instances_open3d.ply", pcd)
