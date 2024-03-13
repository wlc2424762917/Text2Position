import torch

# 加载模型
model_path = "/home/planner/wlc/Text2Position/checkpoints/k360_30-10_scG_pd10_pc4_spY_all/coarse_contN_acc0.49_lrNone_ecl1_eco1_p256_npa1_nca0_f-all_cell_enc_use_query_statedict.pth"
state_dict = torch.load(model_path, map_location=torch.device('cpu'))

# 输出state dict的键
print("Keys in the state dict:")
for key in state_dict:
    print(key)
