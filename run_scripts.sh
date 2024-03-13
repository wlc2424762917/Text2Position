CUDA_VISIBLE_DEVICES=2 nohup python -m training.coarse --batch_size 64 --learning_rate 0.001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --ranking_loss CLIP >> log_clip_loss.txt &

CUDA_VISIBLE_DEVICES=1 python -m training.coarse --batch_size 64 --learning_rate 0.001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --no_pc_augment


CUDA_VISIBLE_DEVICES=1 python -m training.coarse --batch_size 64 --learning_rate 0.001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --no_pc_augment --class_embed

# merge class indice embedding with point feature embedding
CUDA_VISIBLE_DEVICES=1 nohup python -m training.coarse --batch_size 64 --learning_rate 0.001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --no_pc_augment --class_embed >> log_pairwiseLoss_classEmbed_noPcAug &

# merge class indice embedding with point feature embedding and use CLIP loss temperature 0.07 no pc augmentation
CUDA_VISIBLE_DEVICES=1 nohup python -m training.coarse --batch_size 64 --learning_rate 0.001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --no_pc_augment --class_embed --ranking_loss CLIP >> log_clipLoss_classEmbed_noPcAug_T0.05 &

# --language_encoder CLIP_text
CUDA_VISIBLE_DEVICES=1 nohup python -m training.coarse --batch_size 64 --learning_rate 0.00005 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --no_pc_augment --class_embed --ranking_loss CLIP  --language_encoder CLIP_text --epochs 48 >> log_clipCoss_classEmbed_clipTextLanguageEncoder_noPcAug_lr5e-5 &
CUDA_VISIBLE_DEVICES=2 nohup python -m training.coarse --batch_size 64 --learning_rate 0.0001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --no_pc_augment --class_embed --ranking_loss CLIP  --language_encoder CLIP_text --epochs 48 >> log_gpu2_clipCoss_classEmbed_clipTextLanguageEncoder_noPcAug_lr1e-4 &

# --pooling attention clip loss
CUDA_VISIBLE_DEVICES=1 nohup python -m training.coarse --batch_size 64 --learning_rate 0.0001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --no_pc_augment --class_embed --ranking_loss CLIP  --language_encoder LSTM --epochs 48 >> log_gpu1_clipCoss_classEmbed_lstm_poolingAttn_noPcAug_lr1e-4 &


# language clip_transformer no_pc_augment class embedding clip loss lr 1e-4
CUDA_VISIBLE_DEVICES=1 nohup python -m training.coarse --batch_size 64 --learning_rate 0.0001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --no_pc_augment --class_embed --ranking_loss CLIP  --language_encoder CLIP_text_transformer --epochs 48 >> log_gpu1_clipCoss_classEmbed_clipTextTransformerx2LanguageEncoder_fixed_noPcAug_lr1e-4 &
CUDA_VISIBLE_DEVICES=1 python -m training.coarse --batch_size 64 --learning_rate 0.0001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --no_pc_augment --class_embed --ranking_loss CLIP  --language_encoder CLIP_text_transformer --epochs 48

# language clip_transformer no_pc_augment class embedding color embedding clip loss lr 1e-4
CUDA_VISIBLE_DEVICES=3 python -m training.coarse --batch_size 64 --learning_rate 0.0001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --no_pc_augment --color_embed --ranking_loss CLIP  --language_encoder CLIP_text_transformer --epochs 24
CUDA_VISIBLE_DEVICES=3 nohup python -m training.coarse --batch_size 64 --learning_rate 0.0001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --no_pc_augment --class_embed --color_embed --ranking_loss CLIP  --language_encoder CLIP_text_transformer --epochs 48 >> log_gpu3_clipCoss_classEmbed_colorEmbed_clipTextTransformerLanguageEncoder_fixed_noPcAug_lr1e-4 &
# dim 512 no class embedding
CUDA_VISIBLE_DEVICES=5 python -m training.coarse --batch_size 64 --learning_rate 0.0001 --embed_dim 512 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --color_embed --ranking_loss CLIP  --language_encoder CLIP_text_transformer --epochs 24

# language T5_transformer no_pc_augment no class embedding clip loss lr 1e-4
CUDA_VISIBLE_DEVICES=1 python -m training.coarse --batch_size 64 --learning_rate 0.0001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --color_embed --ranking_loss CLIP  --language_encoder T5_text_transformer --epochs 24
# 0.00002
CUDA_VISIBLE_DEVICES=6 python -m training.coarse --batch_size 64 --learning_rate 0.00002 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --color_embed --ranking_loss CLIP  --language_encoder T5_text_transformer --epochs 24 --no_pc_augment

CUDA_VISIBLE_DEVICES=5 python -m training.coarse --batch_size 64 --learning_rate 0.0001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --color_embed --ranking_loss CLIP  --language_encoder T5_text_transformer --epochs 24


# language T5_transformer no_pc_augment class embedding clip loss lr 1e-3
CUDA_VISIBLE_DEVICES=1 python -m training.coarse --batch_size 64 --learning_rate 0.001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --color_embed --class_embed --ranking_loss CLIP  --language_encoder T5_text_transformer --epochs 24 --no_pc_augment
CUDA_VISIBLE_DEVICES=1 python -m training.coarse --batch_size 24 --learning_rate 0.001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --color_embed --class_embed --ranking_loss CLIP  --language_encoder T5_text_transformer --epochs 24 --no_pc_augment


# with augmentation
CUDA_VISIBLE_DEVICES=3 nohup python -m training.coarse --batch_size 64 --learning_rate 0.0001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --class_embed --color_embed --ranking_loss CLIP  --language_encoder CLIP_text_transformer --epochs 48 >> log_gpu3_clipCoss_classEmbed_colorEmbed_clipTextTransformerLanguageEncoder_fixed_with_PcAug_lr1e-4 &

# language clip_transformer with_pc_augment class embedding clip loss lr 1e-4
CUDA_VISIBLE_DEVICES=2 nohup python -m training.coarse --batch_size 64 --learning_rate 0.0001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --class_embed --ranking_loss CLIP  --language_encoder CLIP_text_transformer --epochs 48 >> log_gpu2_clipCoss_classEmbed_clipTextTransformerx2LanguageEncoder_fixed_withPcAug_lr1e-4 &
CUDA_VISIBLE_DEVICES=2 python -m training.coarse --batch_size 64 --learning_rate 0.0001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --class_embed --ranking_loss CLIP  --language_encoder CLIP_text_transformer --epochs 48

# language clip_transformer no_pc_augment class embedding clip loss lr 1e-4 relation transformer
CUDA_VISIBLE_DEVICES=0 python -m training.coarse --batch_size 64 --learning_rate 0.0001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --no_pc_augment --class_embed --color_embed --ranking_loss CLIP  --language_encoder CLIP_text_transformer --use_relation_transformer --epochs 36

# language clip_transformer no_pc_augment no class embedding clip loss lr 1e-4 relation transformer
CUDA_VISIBLE_DEVICES=0 python -m training.coarse --batch_size 64 --learning_rate 0.0001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --no_pc_augment --color_embed --ranking_loss CLIP  --language_encoder CLIP_text_transformer --use_relation_transformer --epochs 36
CUDA_VISIBLE_DEVICES=0 python -m training.coarse --batch_size 4 --learning_rate 0.0001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --no_pc_augment --color_embed --ranking_loss CLIP  --language_encoder CLIP_text_transformer --use_relation_transformer --epochs 36


# language clip_transformer no_pc_augment class embedding clip loss lr 1e-4 clip_2D_text_feature no pointnet
CUDA_VISIBLE_DEVICES=3 python -m training.coarse --batch_size 64 --learning_rate 0.0001 --embed_dim 256 --shuffle --base_path ./data/k360_text_30-10_gridCells_pd10_pc4_shiftPoses_all_nm-6/ --no_pc_augment --color_embed --ranking_loss CLIP  --language_encoder CLIP_text_transformer --only_clip_semantic_feature --use_features color position --epochs 24
# language clip_transformer no_pc_augment class embedding clip loss lr 1e-4 clip_2D_text_feature + pointnet
CUDA_VISIBLE_DEVICES=3 python -m training.coarse --batch_size 64 --learning_rate 0.0001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all_clip_text/ --no_pc_augment --color_embed --ranking_loss CLIP  --language_encoder CLIP_text_transformer --use_clip_semantic_feature --epochs 36


# use semhead
CUDA_VISIBLE_DEVICES=4 python -m training.coarse --batch_size 64 --learning_rate 0.0001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --no_pc_augment --color_embed --ranking_loss CLIP  --language_encoder CLIP_text_transformer --use_semantic_head --use_features class color position --epochs 36


CUDA_VISIBLE_DEVICES=7 python -m training.coarse --batch_size 64 --learning_rate 0.0001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all_clip_text/ --no_pc_augment --color_embed --ranking_loss CLIP  --language_encoder CLIP_text_transformer --only_clip_semantic_feature --use_features color position --epochs 36



# language clip_transformer no_pc_augment  clip loss lr 1e-4 no class embedding
CUDA_VISIBLE_DEVICES=6 python -m training.coarse --batch_size 64 --learning_rate 0.0001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --no_pc_augment --color_embed --ranking_loss CLIP  --language_encoder CLIP_text_transformer  --epochs 36



# lstm clip loss + class embedding + no pc augment
CUDA_VISIBLE_DEVICES=2 nohup python -m training.coarse --batch_size 64 --learning_rate 0.0001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --no_pc_augment --class_embed --ranking_loss CLIP  --language_encoder LSTM --use_edge_conv --epochs 48 >> log_gpu2_clipCoss_classEmbed_LSTMLanguageEncoder_noPcAug_lr1e-4 &


CUDA_VISIBLE_DEVICES=1 nohup python -m training.coarse --batch_size 64 --learning_rate 0.001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --no_pc_augment >> log_no_pc_augment &

python -m training.coarse --batch_size 64 --learning_rate 0.0001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --no_pc_augment --class_embed --ranking_loss CLIP  --language_encoder CLIP_text --epochs 48




CUDA_VISIBLE_DEVICES=6 python -m training.coarse --batch_size 64 --learning_rate 0.00002 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --no_pc_augment --class_embed --color_embed --ranking_loss CLIP  --language_encoder CLIP_text_transformer --clip_text_freeze False --epochs 36


# fine stage
CUDA_VISIBLE_DEVICES=5 python -m training.fine --batch_size 32 --learning_rate 0.0003 --embed_dim 128 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/
CUDA_VISIBLE_DEVICES=1 nohup python -m training.fine --batch_size 32 --learning_rate 0.0003 --embed_dim 128 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ >>log_fine_baseline &


# pointnet pretrain
CUDA_VISIBLE_DEVICES=6 python -m training.pointcloud.pointnet2 --batch_size 64 --learning_rate 0.0001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --no_pc_augment --color_embed --ranking_loss CLIP  --language_encoder CLIP_text_transformer --only_clip_semantic_feature --use_features color position --epochs 36

# main clip feature
CUDA_VISIBLE_DEVICES=5 python -m training.coarse --batch_size 64 --learning_rate 0.0001 --embed_dim 256 --shuffle --base_path /home/wanglichao/Text2Position/data/k360_30-10_scG_pd10_pc4_spY_all_clip_feature/ --no_pc_augment --color_embed --ranking_loss CLIP  --language_encoder CLIP_text_transformer --only_clip_semantic_feature --use_features color position --epochs 24
CUDA_VISIBLE_DEVICES=0 python -m training.coarse --batch_size 64 --learning_rate 0.0001 --embed_dim 256 --shuffle --base_path /home/wanglichao/Text2Position/data/k360_30-10_scG_pd10_pc4_spY_all_clip_feature/ --no_pc_augment --color_embed --ranking_loss CLIP  --language_encoder CLIP_text_transformer --only_clip_semantic_feature --use_features color position --use_relation_transformer --epochs 24

# cell encoder
CUDA_VISIBLE_DEVICES=0 python -m training.coarse --batch_size 64 --learning_rate 0.0001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --color_embed --ranking_loss CLIP  --language_encoder T5_text_transformer --epochs 24 --dataset K360_cell --no_objects
CUDA_VISIBLE_DEVICES=6 python -m training.coarse --batch_size 36 --learning_rate 0.0001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --color_embed --ranking_loss CLIP  --language_encoder T5_text_transformer --epochs 24 --dataset K360_cell --no_objects
CUDA_VISIBLE_DEVICES=4 python -m training.coarse --batch_size 32 --learning_rate 0.001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --color_embed --ranking_loss CLIP  --language_encoder T5_text_transformer --epochs 24 --dataset K360_cell --no_objects
CUDA_VISIBLE_DEVICES=0 python -m training.coarse --batch_size 32 --learning_rate 0.001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --color_embed --ranking_loss CLIP  --language_encoder T5_text_transformer --epochs 24 --dataset K360_cell --no_objects
CUDA_VISIBLE_DEVICES=0 python -m training.coarse --batch_size 32 --learning_rate 0.001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --color_embed --ranking_loss CLIP  --language_encoder T5_text_transformer --epochs 24 --dataset K360_cell --no_objects --color_embed --class_embed  --use_relation_transformer
CUDA_VISIBLE_DEVICES=0 python -m training.coarse --batch_size 4 --learning_rate 0.001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --color_embed --ranking_loss CLIP  --language_encoder T5_text_transformer --epochs 24 --dataset K360_cell --no_objects --color_embed --class_embed --use_relation_transformer
CUDA_VISIBLE_DEVICES=1 python -m training.coarse --batch_size 1 --learning_rate 0.001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --color_embed --ranking_loss CLIP  --language_encoder T5_text_transformer --epochs 24 --dataset K360_cell --no_objects --color_embed --class_embed --use_relation_transformer
CUDA_VISIBLE_DEVICES=0 python -m training.coarse --batch_size 24 --learning_rate 0.001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --color_embed --ranking_loss CLIP  --language_encoder T5_text_transformer --epochs 24 --dataset K360_cell --no_objects --color_embed --class_embed --use_relation_transformer --freeze_cell_encoder

CUDA_VISIBLE_DEVICES=4 python -m training.coarse --batch_size 4 --learning_rate 0.001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --color_embed --ranking_loss CLIP  --language_encoder T5_text_transformer --epochs 24 --dataset K360_cell --no_objects --color_embed --class_embed --use_relation_transformer
CUDA_VISIBLE_DEVICES=6 python -m training.coarse --batch_size 64 --learning_rate 0.001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all_clip_feature/ --ranking_loss CLIP  --language_encoder T5_text_transformer --epochs 24 --dataset K360_cell --no_objects --color_embed --class_embed --use_relation_transformer --freeze_cell_encoder --no_pc_augment

CUDA_VISIBLE_DEVICES=7 python -m training.coarse --batch_size 64 --learning_rate 0.001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all_clip_feature/ --ranking_loss CLIP  --language_encoder T5_text_transformer --epochs 24 --dataset K360_cell --no_objects --color_embed --class_embed --use_relation_transformer --freeze_cell_encoder --use_queries --no_pc_augment
CUDA_VISIBLE_DEVICES=7 python -m training.coarse --batch_size 64 --learning_rate 0.001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all_clip_feature/ --ranking_loss CLIP  --language_encoder T5_text_transformer --epochs 24 --dataset K360_cell --no_objects --color_embed --class_embed --use_relation_transformer --freeze_cell_encoder --no_pc_augment --mask3d_ckpt /home/planner/wlc/Text2Position/checkpoints/k360_30-10_scG_pd10_pc4_spY_all/coarse_contN_acc0.38_lrNone_ecl1_eco1_p256_npa1_nca0_f-all_no_gt_instance_True_query_False_state_dict.pth


CUDA_VISIBLE_DEVICES=0 python -m training.coarse --batch_size 4 --learning_rate 0.001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --color_embed --ranking_loss CLIP  --language_encoder T5_text_transformer --epochs 24 --dataset K360_cell --no_objects --color_embed --class_embed --use_relation_transformer --no_pc_augment

CUDA_VISIBLE_DEVICES=0 python -m training.coarse --batch_size 4 --learning_rate 0.001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --ranking_loss CLIP  --language_encoder T5_text_transformer --epochs 24 --dataset K360_cell --no_objects --color_embed --use_relation_transformer --no_pc_augment --use_queries --two_optimizers

CUDA_VISIBLE_DEVICES=6 python -m training.coarse --batch_size 48 --learning_rate 0.00001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --ranking_loss CLIP  --language_encoder T5_text_transformer --epochs 24 --dataset K360_cell --no_objects --color_embed --class_embed --use_relation_transformer --no_pc_augment --use_queries

CUDA_VISIBLE_DEVICES=0 python -m training.coarse --batch_size 12 --learning_rate 0.001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --ranking_loss CLIP  --language_encoder T5_text_transformer --epochs 24 --dataset K360_cell --no_objects --color_embed --use_relation_transformer --no_pc_augment --use_queries --two_optimizers
CUDA_VISIBLE_DEVICES=0 python -m training.coarse --batch_size 12 --learning_rate 0.001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --ranking_loss CLIP  --language_encoder T5_text_transformer --epochs 24 --dataset K360_cell --no_objects --color_embed --class_embed --use_relation_transformer --no_pc_augment --two_optimizers

CUDA_VISIBLE_DEVICES=1 python -m training.coarse --batch_size 32 --learning_rate 0.001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --ranking_loss CLIP  --language_encoder T5_text_transformer --epochs 24 --dataset K360_cell --no_objects --color_embed --class_embed --use_relation_transformer --no_pc_augment --freeze_cell_encoder

# train freeze cell encoder no query (teacher for query embedding)
CUDA_VISIBLE_DEVICES=7 python -m training.coarse --batch_size 64 --learning_rate 0.001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --ranking_loss CLIP  --language_encoder T5_text_transformer --epochs 24 --dataset K360_cell --no_objects --color_embed --class_embed --use_relation_transformer --freeze_cell_encoder --no_pc_augment

# query embed distillation
CUDA_VISIBLE_DEVICES=4 python -m training.coarse --batch_size 4 --learning_rate 0.001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --ranking_loss CLIP  --language_encoder T5_text_transformer --epochs 24 --dataset K360_cell --no_objects --color_embed --class_embed --use_relation_transformer --freeze_cell_encoder --no_pc_augment --distill_query_embedding --no_pc_augment

# remember to change the mask3d no offset (get instance mask and cell retrieval part)
# change mask3d no offset 33
CUDA_VISIBLE_DEVICES=1 python -m training.coarse --batch_size 2 --learning_rate 0.001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --ranking_loss CLIP  --language_encoder T5_text_transformer --epochs 24 --dataset K360_cell --no_objects --color_embed --class_embed --use_relation_transformer --freeze_cell_encoder --no_pc_augment --mask3d_ckpt /home/wanglichao/Mask3D/saved/t2p_015_q24_new/last-epoch.ckpt
# change mask3d no offset planner
CUDA_VISIBLE_DEVICES=7 python -m training.coarse --batch_size 64 --learning_rate 0.001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --ranking_loss CLIP  --language_encoder T5_text_transformer --epochs 24 --dataset K360_cell --no_objects --color_embed --class_embed --use_relation_transformer --freeze_cell_encoder --no_pc_augment --mask3d_ckpt /home/planner/wlc/t2p_015_q24_new/last.ckpt --no_offset_idx
CUDA_VISIBLE_DEVICES=4 python -m training.coarse --batch_size 2 --learning_rate 0.001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --ranking_loss CLIP  --language_encoder T5_text_transformer --epochs 24 --dataset K360_cell --no_objects --color_embed --class_embed --use_relation_transformer --freeze_cell_encoder --no_pc_augment --mask3d_ckpt /home/planner/wlc/t2p_015_q24_new/last.ckpt --no_offset_idx

CUDA_VISIBLE_DEVICES=4 python -m training.coarse --batch_size 64 --learning_rate 0.001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --ranking_loss CLIP  --language_encoder T5_text_transformer --epochs 24 --dataset K360_cell --no_objects --color_embed --class_embed --use_relation_transformer --freeze_cell_encoder --no_pc_augment --mask3d_ckpt /home/planner/wlc/t2p_015_q24_55.8+/last.ckpt --no_offset_idx
CUDA_VISIBLE_DEVICES=7 python -m training.coarse --batch_size 64 --learning_rate 0.001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --ranking_loss CLIP  --language_encoder T5_text_transformer --epochs 24 --dataset K360_cell --no_objects --color_embed --class_embed --use_relation_transformer --freeze_cell_encoder --no_pc_augment --mask3d_ckpt /home/planner/wlc/t2p_015_q24_55.8+/last.ckpt

CUDA_VISIBLE_DEVICES=4 python -m training.coarse --batch_size 64 --learning_rate 0.001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --ranking_loss CLIP  --language_encoder T5_text_transformer --epochs 24 --dataset K360_cell --no_objects --color_embed --class_embed --use_relation_transformer --freeze_cell_encoder --no_pc_augment --mask3d_ckpt /home/planner/wlc/t2p_015_q24_new/last.ckpt


# no objects no query
CUDA_VISIBLE_DEVICES=6 python -m training.coarse --batch_size 64 --learning_rate 0.001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --ranking_loss CLIP  --language_encoder T5_text_transformer --epochs 24 --dataset K360_cell --no_objects --color_embed --class_embed --use_relation_transformer --freeze_cell_encoder --no_pc_augment --continue_path /home/planner/wlc/Text2Position/checkpoints/k360_30-10_scG_pd10_pc4_spY_all/coarse_contN_acc0.38_lrNone_ecl1_eco1_p256_npa1_nca0_f-all_no_gt_instance_True_query_False_state_dict.pth

# test
CUDA_VISIBLE_DEVICES=0 python -m training.coarse --batch_size 2 --learning_rate 0.001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --ranking_loss CLIP  --language_encoder T5_text_transformer --epochs 24 --dataset K360_cell --no_objects --color_embed --class_embed --use_relation_transformer --freeze_cell_encoder --no_pc_augment --distill_query_embedding

# teacher model
CUDA_VISIBLE_DEVICES=0 python -m training.coarse --batch_size 24 --learning_rate 0.001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --ranking_loss CLIP  --language_encoder T5_text_transformer --epochs 24 --dataset K360_cell --no_objects --color_embed --class_embed --use_relation_transformer --freeze_cell_encoder --no_pc_augment --teacher_model
CUDA_VISIBLE_DEVICES=4 python -m training.coarse --batch_size 64 --learning_rate 0.001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --ranking_loss CLIP  --language_encoder T5_text_transformer --epochs 24 --dataset K360_cell --no_objects --color_embed --class_embed --use_relation_transformer --freeze_cell_encoder --no_pc_augment --teacher_model
# teacher model continue
CUDA_VISIBLE_DEVICES=7 python -m training.coarse --batch_size 64 --learning_rate 0.001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --ranking_loss CLIP  --language_encoder T5_text_transformer --epochs 24 --dataset K360_cell --no_objects --color_embed --class_embed --use_relation_transformer --freeze_cell_encoder --no_pc_augment --teacher_model --continue_path /home/planner/wlc/Text2Position/checkpoints/k360_30-10_scG_pd10_pc4_spY_all/coarse_contN_acc0.58_lrNone_ecl1_eco1_p256_npa1_nca0_f-all_no_gt_instance_True_query_False_teacher_state_dict.pth


python -m evaluation.pipeline --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all_clip_feature/ \
--path_coarse ./checkpoints/k360_30-10_scG_pd10_pc4_spY_all_clip_feature/coarse_contN_acc0.66_lrNone_ecl0_eco1_p256_npa1_nca0_f-color-position_attn.pth \
--path_fine ./checkpoints/fine_acc0.88_lr1_obj-6-16_p256.pth

python -m evaluation.pipeline --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all_clip_feature/ \
--path_coarse /home/wanglichao/Text2Position/checkpoints/k360_30-10_scG_pd10_pc4_spY_all_clip_feature/coarse_contN_acc0.75_lrNone_ecl0_eco1_p256_npa1_nca0_f-color-position_relation.pth \
--path_fine ./checkpoints/fine_acc0.88_lr1_obj-6-16_p256.pth

python -m evaluation.pipeline --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all_clip_feature/ \
--path_coarse /home/wanglichao/Text2Position/checkpoints/k360_30-10_scG_pd10_pc4_spY_all_clip_feature/coarse_contN_acc0.75_lrNone_ecl0_eco1_p256_npa1_nca0_f-color-position_relation.pth \
--path_fine ./checkpoints/fine_acc0.88_lr1_obj-6-16_p256.pth --use_test_set

CUDA_VISIBLE_DEVICES=5 python -m evaluation.pipeline --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all_clip_feature/ \
--path_coarse /home/wanglichao/Text2Position/checkpoints/k360_30-10_scG_pd10_pc4_spY_all_clip_feature/coarse_contN_acc0.75_lrNone_ecl0_eco1_p256_npa1_nca0_f-color-position_relation.pth \
--path_fine ./checkpoints/fine_acc0.88_lr1_obj-6-16_p256.pth --use_alt_descriptions


CUDA_VISIBLE_DEVICES=5 python -m evaluation.pipeline --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all_clip_feature/ \
--path_coarse /home/wanglichao/Text2Position/checkpoints/k360_30-10_scG_pd10_pc4_spY_all/coarse_contN_acc0.54_lrNone_ecl0_eco1_p256_npa1_nca0_f-all.pth \
--path_fine ./checkpoints/fine_acc0.88_lr1_obj-6-16_p256.pth --use_alt_descriptions

CUDA_VISIBLE_DEVICES=5 python -m evaluation.pipeline --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all_clip_feature/ \
--path_coarse /home/wanglichao/Text2Position/checkpoints/k360_30-10_scG_pd10_pc4_spY_all/coarse_contN_acc0.54_lrNone_ecl0_eco1_p256_npa1_nca0_f-all.pth \
--path_fine ./checkpoints/fine_acc0.88_lr1_obj-6-16_p256.pth

CUDA_VISIBLE_DEVICES=5 python -m evaluation.pipeline --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ \
--path_coarse /home/wanglichao/checkpoints_31/k360_30-10_scG_pd10_pc4_spY_all/coarse_contN_acc0.54_lrNone_ecl0_eco1_p256_npa1_nca0_f-all.pth \
--path_fine /home/wanglichao/Text2Position/checkpoints/k360_30-10_scG_pd10_pc4_spY_all_clip_feature/fine_contN_acc0.88_lrNone_obj-6-16_ecl0_eco0_p256_npa1_nca0_f-color-position_only_matcher.pth --use_test_set --plot_retrievals

CUDA_VISIBLE_DEVICES=3 python -m training.fine --batch_size 32 --no_pc_augment --learning_rate 0.0003 --embed_dim 128 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/
CUDA_VISIBLE_DEVICES=3 python -m training.fine --batch_size 32 --learning_rate 0.0003 --embed_dim 128 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/
CUDA_VISIBLE_DEVICES=3 python -m training.fine --batch_size 1 --learning_rate 0.0003 --embed_dim 128 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/

CUDA_VISIBLE_DEVICES=5 python -m training.fine --batch_size 32 --learning_rate 0.0003 --embed_dim 128 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/
CUDA_VISIBLE_DEVICES=3 python -m training.fine --batch_size 32 --no_pc_augment --learning_rate 0.0003 --embed_dim 128 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all_clip_feature/ --language_encoder T5

CUDA_VISIBLE_DEVICES=6 python -m training.fine --batch_size 32 --learning_rate 0.0003 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --no_superglue --language_encoder T5 --use_cross_attention --no_pc_augment
CUDA_VISIBLE_DEVICES=0 python -m training.fine --batch_size 1 --learning_rate 0.0003 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --no_superglue --language_encoder T5 --use_cross_attention --no_pc_augment --no_objects
CUDA_VISIBLE_DEVICES=0 python -m training.fine --batch_size 32 --learning_rate 0.0003 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --no_superglue --language_encoder T5 --use_cross_attention --no_pc_augment --no_objects


# clip semantic feature multi-task
CUDA_VISIBLE_DEVICES=3 python -m training.fine --batch_size 32 --no_pc_augment --learning_rate 0.0003 --embed_dim 128 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all_clip_feature/ --language_encoder CLIP_text --only_clip_semantic_feature --use_features color position

# only matcher
CUDA_VISIBLE_DEVICES=7 python -m training.fine --batch_size 32 --no_pc_augment --learning_rate 0.0003 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all_clip_feature/ --language_encoder CLIP_text --only_clip_semantic_feature --use_features color position --only_matcher --epochs 48
# continue training
CUDA_VISIBLE_DEVICES=7 python -m training.fine --batch_size 32 --no_pc_augment --learning_rate 0.0003 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all_clip_feature/ --language_encoder CLIP_text --only_clip_semantic_feature --use_features color position --epochs 48 --continue_path /home/wanglichao/Text2Position/checkpoints/k360_30-10_scG_pd10_pc4_spY_all_clip_feature/fine_contN_acc0.86_lrNone_obj-6-16_ecl0_eco0_p256_npa1_nca0_f-color-position.pth

CUDA_VISIBLE_DEVICES=6 python -m training.fine --batch_size 32 --no_pc_augment --learning_rate 0.0003 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all_clip_feature/ --language_encoder T5 --only_clip_semantic_feature --use_features color position --only_matcher --epochs 48 --use_cross_attention

CUDA_VISIBLE_DEVICES=7 python -m training.fine --batch_size 32 --no_pc_augment --learning_rate 0.0003 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all_clip_feature/ --language_encoder CLIP_text --only_clip_semantic_feature --use_features color position --epochs 48 --use_cross_attention

# lstm matcher
CUDA_VISIBLE_DEVICES=2 python -m training.fine --batch_size 32 --no_pc_augment --learning_rate 0.0003 --embed_dim 128 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --only_matcher

# clip+t5
CUDA_VISIBLE_DEVICES=4 python -m training.fine --batch_size 32 --no_pc_augment --learning_rate 0.0003 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all_clip_feature/ --language_encoder T5_and_CLIP_text --only_clip_semantic_feature --use_features color position --epochs 48 --use_cross_attention



CUDA_VISIBLE_DEVICES=5 python -m training.fine --batch_size 64 --no_pc_augment --learning_rate 0.00003 --embed_dim 512 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all_clip_feature/ --language_encoder CLIP_text --use_clip_semantic_feature --epochs 36 --use_cross_attention
# fine clip cross (+point net)

CUDA_VISIBLE_DEVICES=7 python -m training.fine --batch_size 64 --no_pc_augment --learning_rate 0.0002 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all_clip_feature/ --language_encoder CLIP_text --only_clip_semantic_feature --use_features color position --only_matcher --epochs 36 --use_cross_attention

# fine continue
CUDA_VISIBLE_DEVICES=7 python -m training.fine --batch_size 32 --no_pc_augment --learning_rate 0.0003 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all_clip_feature/ --language_encoder CLIP_text --only_clip_semantic_feature --use_features color position --epochs 48 --continue_path /home/wanglichao/Text2Position/checkpoints/k360_30-10_scG_pd10_pc4_spY_all_clip_feature/fine_contN_acc0.88_lrNone_obj-6-16_ecl0_eco0_p256_npa1_nca0_f-color-position_only_matcher.pth --use_cross_attention

CUDA_VISIBLE_DEVICES=6 python -m training.fine --batch_size 32 --learning_rate 0.00003 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --language_encoder T5 --epochs 36 --use_cross_attention

CUDA_VISIBLE_DEVICES=1 python -m training.fine --batch_size 64 --no_pc_augment --learning_rate 0.00003 --embed_dim 512 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all_clip_feature/ --language_encoder CLIP_text --use_clip_semantic_feature --epochs 36 --use_cross_attention


# NO OBJECTS EVAL
CUDA_VISIBLE_DEVICES=0 python -m evaluation.pipeline_cell --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ \
--path_coarse  /home/planner/wlc/Text2Position/checkpoints/k360_30-10_scG_pd10_pc4_spY_all_clip_feature/coarse_contN_acc0.46_lrNone_ecl1_eco1_p256_npa1_nca0_f-all_cell_enc_queries.pth \
--path_fine /home/planner/wlc/Text2Position/checkpoints/k360_30-10_scG_pd10_pc4_spY_all/fine_contN_acc0.13_lrNone_obj-6-16_ecl1_eco1_p256_npa1_nca0_f-all.pth \
--ranking_loss CLIP  --language_encoder T5_text_transformer --no_objects --color_embed --class_embed --use_relation_transformer --freeze_cell_encoder --embed_dim 256 --use_queries --no_pc_augment --dataset K360_cell --use_test_set --plot_retrievals