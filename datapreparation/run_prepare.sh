
CUDA_VISIBLE_DEVICES=7 python -m datapreparation.kitti360pose.prepare --scene_name 2013_05_28_drive_0000_sync --path_in /home/wanglichao/KITTI-360 --path_out ./data/k360_new --cell_size 30 --cell_dist 10 --pose_dist 10 --pose_count 4 --shift_poses --grid_cells
python -m datapreparation.kitti360pose.prepare --scene_name 2013_05_28_drive_0000_sync --path_in /home/wanglichao/KITTI-360 --path_out ./data/k360_new --cell_size 30 --cell_dist 10 --pose_dist 10 --pose_count 4 --shift_poses --grid_cells