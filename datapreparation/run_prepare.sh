
CUDA_VISIBLE_DEVICES=1 python -m datapreparation.kitti360pose.prepare --scene_name 2013_05_28_drive_0000_sync --path_in /home/wanglichao/KITTI-360 --path_out ./data/k360_new --cell_size 30 --cell_dist 10 --pose_dist 10 --pose_count 4 --shift_poses --grid_cells
CUDA_VISIBLE_DEVICES=1 nohup python -m datapreparation.kitti360pose.prepare --scene_name 2013_05_28_drive_0000_sync --path_in /home/wanglichao/KITTI-360 --path_out ./data/k360_new --cell_size 30 --cell_dist 10 --pose_dist 10 --pose_count 4 --shift_poses --grid_cells >> log_k360_new_0000.txt &

CUDA_VISIBLE_DEVICES=1 python -m datapreparation.kitti360pose.prepare --scene_name 2013_05_28_drive_0002_sync --path_in /home/wanglichao/KITTI-360 --path_out ./data/k360_new --cell_size 30 --cell_dist 10 --pose_dist 10 --pose_count 4 --shift_poses --grid_cells
CUDA_VISIBLE_DEVICES=6 nohup python -m datapreparation.kitti360pose.prepare --scene_name 2013_05_28_drive_0002_sync --path_in /home/wanglichao/KITTI-360 --path_out ./data/k360_new --cell_size 30 --cell_dist 10 --pose_dist 10 --pose_count 4 --shift_poses --grid_cells >> log_k360_new_0002.txt &
CUDA_VISIBLE_DEVICES=2 nohup python -m datapreparation.kitti360pose.prepare --scene_name 2013_05_28_drive_0003_sync --path_in /home/wanglichao/KITTI-360 --path_out ./data/k360_new --cell_size 30 --cell_dist 10 --pose_dist 10 --pose_count 4 --shift_poses --grid_cells >> log_k360_new_0003.txt &

CUDA_VISIBLE_DEVICES=1 python -m datapreparation.kitti360pose.prepare_no_sam --scene_name 2013_05_28_drive_0010_sync --path_in /home/wanglichao/KITTI-360 --path_out ./data/k360_new --cell_size 30 --cell_dist 10 --pose_dist 10 --pose_count 4 --shift_poses --grid_cells
CUDA_VISIBLE_DEVICES=1 nohup python -m datapreparation.kitti360pose.prepare_no_sam --scene_name 2013_05_28_drive_0010_sync --path_in /home/wanglichao/KITTI-360 --path_out ./data/k360_new --cell_size 30 --cell_dist 10 --pose_dist 10 --pose_count 4 --shift_poses --grid_cells >> log_k360_new_0010_test_obj_proj_num.txt &

CUDA_VISIBLE_DEVICES=5 python -m datapreparation.kitti360pose.prepare --scene_name 2013_05_28_drive_0010_sync --path_in /home/wanglichao/KITTI-360 --path_out ./data/k360_new --cell_size 30 --cell_dist 10 --pose_dist 10 --pose_count 4 --shift_poses --grid_cells
CUDA_VISIBLE_DEVICES=4 nohup python -m datapreparation.kitti360pose.prepare --scene_name 2013_05_28_drive_0010_sync --path_in /home/wanglichao/KITTI-360 --path_out ./data/k360_new --cell_size 30 --cell_dist 10 --pose_dist 10 --pose_count 4 --shift_poses --grid_cells >> log_k360_new_0010.txt &


## sever 31
CUDA_VISIBLE_DEVICES=1 python -m datapreparation.kitti360pose.prepare --scene_name 2013_05_28_drive_0004_sync --path_in /home/wanglichao/KITTI-360 --path_out ./data/k360_new --cell_size 30 --cell_dist 10 --pose_dist 10 --pose_count 4 --shift_poses --grid_cells
CUDA_VISIBLE_DEVICES=1 nohup python -m datapreparation.kitti360pose.prepare --scene_name 2013_05_28_drive_0004_sync --path_in /home/wanglichao/KITTI-360 --path_out ./data/k360_new --cell_size 30 --cell_dist 10 --pose_dist 10 --pose_count 4 --shift_poses --grid_cells >> log_k360_new_0004.txt &

CUDA_VISIBLE_DEVICES=1 python -m datapreparation.kitti360pose.prepare --scene_name 2013_05_28_drive_0005_sync --path_in /home/wanglichao/KITTI-360 --path_out ./data/k360_new --cell_size 30 --cell_dist 10 --pose_dist 10 --pose_count 4 --shift_poses --grid_cells
CUDA_VISIBLE_DEVICES=2 nohup python -m datapreparation.kitti360pose.prepare --scene_name 2013_05_28_drive_0005_sync --path_in /home/wanglichao/KITTI-360 --path_out ./data/k360_new --cell_size 30 --cell_dist 10 --pose_dist 10 --pose_count 4 --shift_poses --grid_cells >> log_k360_new_0005.txt &


CUDA_VISIBLE_DEVICES=3 python -m datapreparation.kitti360pose.prepare --scene_name 2013_05_28_drive_0006_sync --path_in /home/wanglichao/KITTI-360 --path_out ./data/k360_new --cell_size 30 --cell_dist 10 --pose_dist 10 --pose_count 4 --shift_poses --grid_cells
CUDA_VISIBLE_DEVICES=6 nohup python -m datapreparation.kitti360pose.prepare --scene_name 2013_05_28_drive_0006_sync --path_in /home/wanglichao/KITTI-360 --path_out ./data/k360_new --cell_size 30 --cell_dist 10 --pose_dist 10 --pose_count 4 --shift_poses --grid_cells >> log_k360_new_0006.txt &

CUDA_VISIBLE_DEVICES=4 python -m datapreparation.kitti360pose.prepare --scene_name 2013_05_28_drive_0007_sync --path_in /home/wanglichao/KITTI-360 --path_out ./data/k360_new --cell_size 30 --cell_dist 10 --pose_dist 10 --pose_count 4 --shift_poses --grid_cells
CUDA_VISIBLE_DEVICES=3 nohup python -m datapreparation.kitti360pose.prepare --scene_name 2013_05_28_drive_0007_sync --path_in /home/wanglichao/KITTI-360 --path_out ./data/k360_new --cell_size 30 --cell_dist 10 --pose_dist 10 --pose_count 4 --shift_poses --grid_cells >> log_k360_new_0007.txt &

CUDA_VISIBLE_DEVICES=4 python -m datapreparation.kitti360pose.prepare --scene_name 2013_05_28_drive_0009_sync --path_in /home/wanglichao/KITTI-360 --path_out ./data/k360_new --cell_size 30 --cell_dist 10 --pose_dist 10 --pose_count 4 --shift_poses --grid_cells
CUDA_VISIBLE_DEVICES=3 nohup python -m datapreparation.kitti360pose.prepare --scene_name 2013_05_28_drive_0009_sync --path_in /home/wanglichao/KITTI-360 --path_out ./data/k360_new --cell_size 30 --cell_dist 10 --pose_dist 10 --pose_count 4 --shift_poses --grid_cells >> log_k360_new_0009.txt &

CUDA_VISIBLE_DEVICES=6 python -m datapreparation.kitti360pose.prepare --scene_name 2013_05_28_drive_0010_sync --path_in /home/wanglichao/KITTI-360 --path_out ./data/k360_new --cell_size 30 --cell_dist 10 --pose_dist 10 --pose_count 4 --shift_poses --grid_cells
CUDA_VISIBLE_DEVICES=6 nohup python -m datapreparation.kitti360pose.prepare --scene_name 2013_05_28_drive_0003_sync --path_in /home/wanglichao/KITTI-360 --path_out ./data/k360_new --cell_size 30 --cell_dist 10 --pose_dist 10 --pose_count 4 --shift_poses --grid_cells >> log_k360_new_0003.txt &

CUDA_VISIBLE_DEVICES=7 python -m datapreparation.kitti360pose.prepare --scene_name 2013_05_28_drive_0002_sync --path_in /home/wanglichao/KITTI-360 --path_out ./data/k360_new --cell_size 30 --cell_dist 10 --pose_dist 10 --pose_count 4 --shift_poses --grid_cells
CUDA_VISIBLE_DEVICES=7 nohup python -m datapreparation.kitti360pose.prepare --scene_name 2013_05_28_drive_0002_sync --path_in /home/wanglichao/KITTI-360 --path_out ./data/k360_new --cell_size 30 --cell_dist 10 --pose_dist 10 --pose_count 4 --shift_poses --grid_cells >> log_k360_new_0002.txt &

CUDA_VISIBLE_DEVICES=7 python -m datapreparation.kitti360pose.prepare --scene_name 2013_05_28_drive_0004_sync --path_in /home/wanglichao/KITTI-360 --path_out ./data/k360_new --cell_size 30 --cell_dist 10 --pose_dist 10 --pose_count 4 --shift_poses --grid_cells
CUDA_VISIBLE_DEVICES=7 nohup python -m datapreparation.kitti360pose.prepare --scene_name 2013_05_28_drive_0004_sync --path_in /home/wanglichao/KITTI-360 --path_out ./data/k360_new --cell_size 30 --cell_dist 10 --pose_dist 10 --pose_count 4 --shift_poses --grid_cells >> log_k360_new_0004.txt &


python -m datapreparation.kitti360pose.prepare --scene_name 2013_05_28_drive_0010_sync --path_in /home/wanglichao/KITTI-360 --path_out ./data/k360_new --cell_size 30 --cell_dist 10 --pose_dist 10 --pose_count 4 --shift_poses --grid_cells

# text
# 34 server prepare changed
CUDA_VISIBLE_DEVICES=0 python -m datapreparation.kitti360pose.prepare --scene_name 2013_05_28_drive_0000_sync --path_in /home/wanglichao/KITTI-360 --path_out ./data/k360_text --cell_size 30 --cell_dist 10 --pose_dist 10 --pose_count 4 --shift_poses --grid_cells
CUDA_VISIBLE_DEVICES=1 python -m datapreparation.kitti360pose.prepare --scene_name 2013_05_28_drive_0002_sync --path_in /home/wanglichao/KITTI-360 --path_out ./data/k360_text --cell_size 30 --cell_dist 10 --pose_dist 10 --pose_count 4 --shift_poses --grid_cells
CUDA_VISIBLE_DEVICES=3 python -m datapreparation.kitti360pose.prepare --scene_name 2013_05_28_drive_0003_sync --path_in /home/wanglichao/KITTI-360 --path_out ./data/k360_text --cell_size 30 --cell_dist 10 --pose_dist 10 --pose_count 4 --shift_poses --grid_cells
CUDA_VISIBLE_DEVICES=1 python -m datapreparation.kitti360pose.prepare --scene_name 2013_05_28_drive_0004_sync --path_in /home/wanglichao/KITTI-360 --path_out ./data/k360_text --cell_size 30 --cell_dist 10 --pose_dist 10 --pose_count 4 --shift_poses --grid_cells
CUDA_VISIBLE_DEVICES=0 python -m datapreparation.kitti360pose.prepare --scene_name 2013_05_28_drive_0005_sync --path_in /home/wanglichao/KITTI-360 --path_out ./data/k360_text --cell_size 30 --cell_dist 10 --pose_dist 10 --pose_count 4 --shift_poses --grid_cells
CUDA_VISIBLE_DEVICES=0 python -m datapreparation.kitti360pose.prepare --scene_name 2013_05_28_drive_0006_sync --path_in /home/wanglichao/KITTI-360 --path_out ./data/k360_text --cell_size 30 --cell_dist 10 --pose_dist 10 --pose_count 4 --shift_poses --grid_cells
CUDA_VISIBLE_DEVICES=0 python -m datapreparation.kitti360pose.prepare --scene_name 2013_05_28_drive_0007_sync --path_in /home/wanglichao/KITTI-360 --path_out ./data/k360_text --cell_size 30 --cell_dist 10 --pose_dist 10 --pose_count 4 --shift_poses --grid_cells
CUDA_VISIBLE_DEVICES=1 python -m datapreparation.kitti360pose.prepare --scene_name 2013_05_28_drive_0009_sync --path_in /home/wanglichao/KITTI-360 --path_out ./data/k360_text --cell_size 30 --cell_dist 10 --pose_dist 10 --pose_count 4 --shift_poses --grid_cells
CUDA_VISIBLE_DEVICES=1 python -m datapreparation.kitti360pose.prepare --scene_name 2013_05_28_drive_0010_sync --path_in /home/wanglichao/KITTI-360 --path_out ./data/k360_text --cell_size 30 --cell_dist 10 --pose_dist 10 --pose_count 4 --shift_poses --grid_cells
