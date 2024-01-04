# # This code is for v1 of the openai package: pypi.org/project/openai
# from openai import OpenAI
# import os
# os.environ["OPENAI_API_KEY"] = "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"
#
# client = OpenAI()
# response = client.chat.completions.create(
#   model="gpt-3.5-turbo",
#   messages=[
#     {
#       "role": "user",
#       "content": ""
#     }
#   ],
#   temperature=1,
#   max_tokens=256,
#   top_p=1,
#   frequency_penalty=0,
#   presence_penalty=0
# )

# import torch
# import clip
# from PIL import Image
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)
#
# # image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
# text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)
# print(text.shape)
# with torch.no_grad():
#     # image_features = model.encode_image(image)
#     # print(image_features.shape)
#     text_features = model.encode_text(text)
#     print(text_features.shape)
#     # logits_per_image, logits_per_text = model(image, text)
#     # probs = logits_per_image.softmax(dim=-1).cpu().numpy()
#
# #print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# class MaxPoolMultiHeadSelfAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads):
#         super(MaxPoolMultiHeadSelfAttention, self).__init__()
#         self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
#         self.num_heads = num_heads
#
#     def forward(self, embeddings, batch):
#         # 计算每个批次的最大长度
#         max_len = max([sum(batch == i) for i in batch.unique()])
#
#         # 初始化填充后的嵌入和掩码
#         padded_embeddings = torch.zeros(len(batch.unique()), max_len, embeddings.size(-1))
#         mask = torch.ones(len(batch.unique()), max_len, max_len)
#
#         # 对每个批次添加填充并创建掩码
#         for i, b in enumerate(batch.unique()):
#             # print(batch == b)
#             # print(embeddings[batch == b].shape)
#             batch_embeddings = embeddings[batch == b]
#             padded_embeddings[i, :batch_embeddings.size(0)] = batch_embeddings
#             mask[i, :batch_embeddings.size(0), :batch_embeddings.size(0)] = 0
#
#         # 调整形状以适应多头注意力模块
#         padded_embeddings = padded_embeddings.transpose(0, 1)  # (max_len, B, D)
#
#         # 重塑掩码以匹配多头注意力的头数
#         mask = mask.repeat(self.num_heads, 1, 1)  # 形状变为 [num_heads * B, max_len, max_len]
#
#         # 应用多头自注意力机制
#         attn_output, _ = self.multihead_attn(padded_embeddings, padded_embeddings, padded_embeddings, attn_mask=mask)
#         attn_output = attn_output.transpose(0, 1)
#
#         # 应用最大池化
#         maxpooled_output = F.max_pool1d(attn_output.transpose(1, 2), kernel_size=attn_output.size(1))
#
#         return maxpooled_output.squeeze()
#
# # 创建模型
# embed_dim = 3
# num_heads = 1
# model = MaxPoolMultiHeadSelfAttention(embed_dim, num_heads)
#
# # 创建随机嵌入和批次索引
# torch.manual_seed(0)  # 为了可重复性
# N = 10  # 总实例数
# D = embed_dim  # 嵌入维度
# B = 3   # 批次数量
#
# # 生成随机嵌入
# embeddings = torch.rand(N, D)
#
# # 生成模拟批次索引
# batch_indices = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 2])
# print("Batch Indices:", batch_indices)
#
# # 使用模型
# output = model(embeddings, batch_indices)
#
# # 打印输出
# print("Output Shape:", output.shape)

import torch
import torch.nn as nn
import torch.nn.functional as F

# class TransformerWithMaxPool(nn.Module):
#     def __init__(self, d_model, nhead, num_layers, dim_feedforward):
#         super(TransformerWithMaxPool, self).__init__()
#         # 创建 Transformer 编码器层
#         encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#         # MaxPooling 层
#         self.maxpool = nn.AdaptiveMaxPool1d(1)
#
#     def forward(self, src):
#         # Transformer 编码器处理
#         transformer_output = self.transformer_encoder(src)
#         # 转换维度以适应 MaxPooling 层
#         transformer_output = transformer_output.permute(1, 2, 0)
#         # 应用 MaxPooling
#         pooled_output = self.maxpool(transformer_output)
#         # 再次转换维度
#         pooled_output = pooled_output.permute(2, 0, 1)
#         return pooled_output.squeeze(0)
#
# # 创建模型实例
# d_model = 512
# nhead = 8
# num_layers = 3
# dim_feedforward = 2048
#
# model = TransformerWithMaxPool(d_model, nhead, num_layers, dim_feedforward)

# # 生成随机输入数据
# seq_len = 10  # 序列长度
# batch_size = 5  # 批处理大小
#
# src = torch.rand(seq_len, batch_size, d_model)  # 随机生成输入数据
#
# # 传递输入数据到模型
# output = model(src)
#
# print(output.shape)

# import numpy as np
# a = np.array([[1,2],[4,5],[3,6]])
# print(a)
# a = a.T
# print(a)
