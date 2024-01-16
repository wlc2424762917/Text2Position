import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F



from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class Custom_Clip_LanguageEncoder(nn.Module):
    def __init__(self, clip_version):
        super(Custom_Clip_LanguageEncoder, self).__init__()
        self.model, _ = clip.load(clip_version)
        self.model.float()
        for name, param in self.model.named_parameters():
            param.requires_grad_(False)
            # print(name, param.requires_grad)
        self.adapter = Adapter(512, 4)
        self.ratio = 0.2

    def forward(self, descriptions):
        text_inputs = torch.cat([clip.tokenize(desc) for desc in descriptions]).to(self.device)
        # text_inputs.to(self.device)# [B, 77]
        text_features = self.model.encode_text(text_inputs)  # [B, 512]
        x = self.adapter(text_features)
        text_features = self.ratio * x + (1 - self.ratio) * text_features
        # text_features = F.normalize(text_features, dim=-1)
        # convert to float32, but dont change device
        text_features = text_features.type(torch.FloatTensor).to(self.device)
        return text_features

    @property
    def device(self):
        return next(self.model.parameters()).device


import torch
import clip
from torch import nn

# 假设 Adapter 和 Custom_Clip_LanguageEncoder 类定义如你所给

# 创建一个 Custom_Clip_LanguageEncoder 实例
# 注意选择一个有效的 CLIP 版本，例如 'ViT-B/32'
clip_version = 'ViT-B/32'
model = Custom_Clip_LanguageEncoder(clip_version)

# 创建一些示例文本数据
descriptions = ["a photo of a cat", "a photo of a dog"]

# 将模型放在正确的设备上
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 测试 forward 方法

text_features = model(descriptions)
print(text_features.shape)

# 检查输出的形状是否正确，例如 [2, 512]，取决于描述的数量和模型的输出维度

