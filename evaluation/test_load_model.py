import torch
path_coarse = "/home/wanglichao/Text2Pos-CVPR2022/checkpoints/coarse_contN_acc0.35_lr1_p256.pth"
path_fine = "/home/wanglichao/Text2Pos-CVPR2022/checkpoints/fine_acc0.88_lr1_obj-6-16_p256.pth"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("device:", device, torch.cuda.get_device_name(0))

model_retrieval = torch.load(path_coarse, map_location=torch.device("cpu"))
model_matching = torch.load(path_fine, map_location=torch.device("cpu"))
model_retrieval.to(device)
model_matching.to(device)