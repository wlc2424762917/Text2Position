CUDA_VISIBLE_DEVICES=2 nohup python -m training.coarse --batch_size 64 --learning_rate 0.001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --ranking_loss CLIP >> log_clip_loss.txt &

CUDA_VISIBLE_DEVICES=1 python -m training.coarse --batch_size 64 --learning_rate 0.001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --no_pc_augment


CUDA_VISIBLE_DEVICES=1 python -m training.coarse --batch_size 64 --learning_rate 0.001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --no_pc_augment --class_embed

# merge class indice embedding with point feature embedding
CUDA_VISIBLE_DEVICES=1 nohup python -m training.coarse --batch_size 64 --learning_rate 0.001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --no_pc_augment --class_embed >> log_pairwiseLoss_classEmbed_noPcAug &

# merge class indice embedding with point feature embedding and use CLIP loss temperature 0.07 no pc augmentation
CUDA_VISIBLE_DEVICES=1 nohup python -m training.coarse --batch_size 64 --learning_rate 0.001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --no_pc_augment --class_embed --ranking_loss CLIP >> log_clipLoss_classEmbed_noPcAug_T0.05 &

# clip loss with t=0.07 and lr=0.0001 no pc augmentation
CUDA_VISIBLE_DEVICES=1 nohup python -m training.coarse --batch_size 64 --learning_rate 0.0001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --ranking_loss CLIP --epochs 36 --no_pc_augment >> log_clip_loss_no_pc_augment &

# --language_encoder CLIP_text
CUDA_VISIBLE_DEVICES=7 nohup python -m training.coarse --batch_size 64 --learning_rate 0.0001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --no_pc_augment --class_embed --ranking_loss CLIP  --language_encoder CLIP_text >> log_clipCoss_classEmbed_clipTextLanguageEncoder_noPcAug &


CUDA_VISIBLE_DEVICES=1 nohup python -m training.coarse --batch_size 64 --learning_rate 0.001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --no_pc_augment >> log_no_pc_augment &

