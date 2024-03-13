# no_offset no_query no_objects
CUDA_VISIBLE_DEVICES=7 python -m training.coarse --batch_size 64 --learning_rate 0.001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --ranking_loss CLIP  --language_encoder T5_text_transformer --epochs 24 --dataset K360_cell --no_objects --color_embed --class_embed --use_relation_transformer --freeze_cell_encoder --no_pc_augment --mask3d_ckpt /home/planner/wlc/t2p_015_q24_new/last.ckpt --no_offset_idx
CUDA_VISIBLE_DEVICES=7 python -m training.coarse --batch_size 64 --learning_rate 0.001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --ranking_loss CLIP  --language_encoder T5_text_transformer --epochs 24 --dataset K360_cell --no_objects --color_embed --class_embed --use_relation_transformer --freeze_cell_encoder --no_pc_augment --continue_path /home/planner/wlc/Text2Position/checkpoints/k360_30-10_scG_pd10_pc4_spY_all/coarse_contY_acc0.37_lrNone_ecl1_eco1_p256_npa1_nca0_f-all_no_gt_instance_True_query_False_state_dict.pth --no_offset_idx


# no_offset use_query no_objects
CUDA_VISIBLE_DEVICES=7 python -m training.coarse --batch_size 64 --learning_rate 0.001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --ranking_loss CLIP  --language_encoder T5_text_transformer --epochs 24 --dataset K360_cell --no_objects --color_embed --class_embed --use_queries --freeze_cell_encoder --no_pc_augment --mask3d_ckpt /home/planner/wlc/t2p_015_q24_new/last.ckpt
CUDA_VISIBLE_DEVICES=6 python -m training.coarse --batch_size 64 --learning_rate 0.001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --ranking_loss CLIP  --language_encoder T5_text_transformer --epochs 24 --dataset K360_cell --no_objects --color_embed --class_embed --use_queries --freeze_cell_encoder --no_pc_augment --continue_path /home/planner/wlc/Text2Position/checkpoints/k360_30-10_scG_pd10_pc4_spY_all/coarse_contY_acc0.47_lrNone_ecl1_eco1_p256_npa1_nca0_f-all_no_gt_instance_True_query_True_state_dict.pth --no_offset_idx

# query embed distillation
CUDA_VISIBLE_DEVICES=4 python -m training.coarse --batch_size 64 --learning_rate 0.001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --ranking_loss CLIP  --language_encoder T5_text_transformer --epochs 24 --dataset K360_cell --no_objects --color_embed --class_embed --use_relation_transformer --freeze_cell_encoder --no_pc_augment --distill_query_embedding --no_pc_augment  --continue_path /home/planner/wlc/Text2Position/checkpoints/k360_30-10_scG_pd10_pc4_spY_all/coarse_contN_acc0.52_lrNone_ecl1_eco1_p256_npa1_nca0_f-all_no_gt_instance_True_query_False_state_dict_no_offset_558.pth


# train teacher model
CUDA_VISIBLE_DEVICES=4 python -m training.coarse --batch_size 64 --learning_rate 0.001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --ranking_loss CLIP  --language_encoder T5_text_transformer --epochs 24 --dataset K360_cell --no_objects --color_embed --class_embed --use_relation_transformer --freeze_cell_encoder --no_pc_augment --teacher_model --mask3d_ckpt /home/planner/wlc/t2p_015_q24_new/last.ckpt --no_offset_idx

# distill cell_feature
CUDA_VISIBLE_DEVICES=0 python -m training.coarse_distill --batch_size 2 --learning_rate 0.001 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --ranking_loss CLIP  --language_encoder T5_text_transformer --epochs 24 --dataset K360_cell --no_objects --color_embed --class_embed --use_relation_transformer --freeze_cell_encoder --distill_cell_feature --no_pc_augment



#### train fine ####
CUDA_VISIBLE_DEVICES=7 python -m training.fine --batch_size 32 --learning_rate 0.0003 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --no_superglue --language_encoder T5 --use_cross_attention --no_pc_augment --no_objects --class_embed --color_embed --freeze_cell_encoder

# not freeze cell encoder
CUDA_VISIBLE_DEVICES=7 python -m training.fine --batch_size 32 --learning_rate 0.0003 --embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ --no_superglue --language_encoder T5 --use_cross_attention --no_pc_augment --no_objects --class_embed --color_embed







###### eval ######

CUDA_VISIBLE_DEVICES=0 python -m evaluation.pipeline_cell --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ \
--path_coarse  /home/planner/wlc/Text2Position/checkpoints/k360_30-10_scG_pd10_pc4_spY_all/coarse_contN_acc0.52_lrNone_ecl1_eco1_p256_npa1_nca0_f-all_no_gt_instance_True_query_False_state_dict_no_offset_558.pth \
--path_fine /home/planner/wlc/Text2Position/checkpoints/k360_30-10_scG_pd10_pc4_spY_all/fine_contN_acc0.13_lrNone_obj-6-16_ecl1_eco1_p256_npa1_nca0_f-all.pth \
--ranking_loss CLIP  --language_encoder T5_text_transformer --no_objects --color_embed --class_embed --use_relation_transformer --freeze_cell_encoder --embed_dim 256 --no_pc_augment --dataset K360_cell --use_test_set