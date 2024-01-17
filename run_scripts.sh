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

# language T5_transformer no_pc_augment no class embedding clip loss lr 1e-4 relation transformer
CUDA_VISIBLE_DEVICES=5 python -m training.coarse --batch_size 32 --learning_rate 0.0001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --color_embed --ranking_loss CLIP  --language_encoder T5_text_transformer --epochs 24

CUDA_VISIBLE_DEVICES=5 python -m training.coarse --batch_size 64 --learning_rate 0.0001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --color_embed --ranking_loss CLIP  --language_encoder T5_text_transformer --epochs 24

# with augmentation
CUDA_VISIBLE_DEVICES=3 nohup python -m training.coarse --batch_size 64 --learning_rate 0.0001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --class_embed --color_embed --ranking_loss CLIP  --language_encoder CLIP_text_transformer --epochs 48 >> log_gpu3_clipCoss_classEmbed_colorEmbed_clipTextTransformerLanguageEncoder_fixed_with_PcAug_lr1e-4 &

# language clip_transformer with_pc_augment class embedding clip loss lr 1e-4
CUDA_VISIBLE_DEVICES=2 nohup python -m training.coarse --batch_size 64 --learning_rate 0.0001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --class_embed --ranking_loss CLIP  --language_encoder CLIP_text_transformer --epochs 48 >> log_gpu2_clipCoss_classEmbed_clipTextTransformerx2LanguageEncoder_fixed_withPcAug_lr1e-4 &
CUDA_VISIBLE_DEVICES=2 python -m training.coarse --batch_size 64 --learning_rate 0.0001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --class_embed --ranking_loss CLIP  --language_encoder CLIP_text_transformer --epochs 48

# language clip_transformer no_pc_augment class embedding clip loss lr 1e-4 relation transformer
CUDA_VISIBLE_DEVICES=0 python -m training.coarse --batch_size 64 --learning_rate 0.0001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --no_pc_augment --class_embed --color_embed --ranking_loss CLIP  --language_encoder CLIP_text_transformer --use_relation_transformer --epochs 36

# language clip_transformer no_pc_augment no class embedding clip loss lr 1e-4 relation transformer
CUDA_VISIBLE_DEVICES=0 python -m training.coarse --batch_size 64 --learning_rate 0.0001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --no_pc_augment --color_embed --ranking_loss CLIP  --language_encoder CLIP_text_transformer --use_relation_transformer --epochs 36


# language clip_transformer no_pc_augment class embedding clip loss lr 1e-4 clip_2D_text_feature no pointnet
CUDA_VISIBLE_DEVICES=3 python -m training.coarse --batch_size 64 --learning_rate 0.0001 --embed_dim 256 --shuffle --base_path ./data/k360_text_30-10_gridCells_pd10_pc4_shiftPoses_all_nm-6/ --no_pc_augment --color_embed --ranking_loss CLIP  --language_encoder CLIP_text_transformer --only_clip_semantic_feature --use_features color position --epochs 36
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
