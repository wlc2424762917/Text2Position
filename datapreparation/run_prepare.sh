
CUDA_VISIBLE_DEVICES=1 python -m datapreparation.kitti360pose.prepare --scene_name 2013_05_28_drive_0000_sync --path_in /home/wanglichao/KITTI-360 --path_out ./data/k360_new --cell_size 30 --cell_dist 10 --pose_dist 10 --pose_count 4 --shift_poses --grid_cells
CUDA_VISIBLE_DEVICES=1 nohup python -m datapreparation.kitti360pose.prepare --scene_name 2013_05_28_drive_0000_sync --path_in /home/wanglichao/KITTI-360 --path_out ./data/k360_new --cell_size 30 --cell_dist 10 --pose_dist 10 --pose_count 4 --shift_poses --grid_cells >> log_k360_new_0000.txt &

CUDA_VISIBLE_DEVICES=3 python -m datapreparation.kitti360pose.prepare --scene_name 2013_05_28_drive_0002_sync --path_in /home/wanglichao/KITTI-360 --path_out ./data/k360_new --cell_size 30 --cell_dist 10 --pose_dist 10 --pose_count 4 --shift_poses --grid_cells
CUDA_VISIBLE_DEVICES=3 nohup python -m datapreparation.kitti360pose.prepare --scene_name 2013_05_28_drive_0002_sync --path_in /home/wanglichao/KITTI-360 --path_out ./data/k360_new --cell_size 30 --cell_dist 10 --pose_dist 10 --pose_count 4 --shift_poses --grid_cells >> log_k360_new_0002.txt &
CUDA_VISIBLE_DEVICES=2 nohup python -m datapreparation.kitti360pose.prepare --scene_name 2013_05_28_drive_0003_sync --path_in /home/wanglichao/KITTI-360 --path_out ./data/k360_new --cell_size 30 --cell_dist 10 --pose_dist 10 --pose_count 4 --shift_poses --grid_cells >> log_k360_new_0003.txt &


python -m datapreparation.kitti360pose.prepare --scene_name 2013_05_28_drive_0000_sync --path_in /home/wanglichao/KITTI-360 --path_out ./data/k360_new --cell_size 30 --cell_dist 10 --pose_dist 10 --pose_count 4 --shift_poses --grid_cells