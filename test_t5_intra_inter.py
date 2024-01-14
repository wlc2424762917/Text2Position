import torch
from torch import nn
from transformers import T5Tokenizer, T5Model

# 步骤 2: 加载T5模型
tokenizer = T5Tokenizer.from_pretrained('/home/wanglichao/Text2Position/flan-t5-base')
t5_model = T5Model.from_pretrained('/home/wanglichao/Text2Position/flan-t5-basel')

# 步骤 3: 定义Intra-Inter Transformer模型
class IntraInterTransformer(nn.Module):
    def __init__(self):
        super(IntraInterTransformer, self).__init__()
        # 定义模型结构，这里只是示意，可以根据需求调整
        self.layer1 = nn.Linear(512, 256)  # 假设输入大小为512
        self.layer2 = nn.Linear(256, 128)

    def forward(self, x):
        x = torch.mean(x, dim=1)  # 简单的平均池化
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x

# 步骤 4: 整合T5和自定义Transformer
def get_hint_embeddings(hints):
    embeddings = []
    for hint in hints:
        input_ids = tokenizer(hint, return_tensors='pt').input_ids
        outputs = t5_model(input_ids=input_ids)
        embeddings.append(outputs.last_hidden_state)
    embeddings = torch.cat(embeddings, dim=1)
    return embeddings

def aggregate_information(hints):
    hint_embeddings = get_hint_embeddings(hints)
    print(f"hint_embeddings.shape: {hint_embeddings.shape}")
    intra_inter_model = IntraInterTransformer()
    aggregated_output = intra_inter_model(hint_embeddings)
    return aggregated_output

# 步骤 5: 测试代码
hints = ["This is the first hint.", "Here is another hint."]
aggregated_output = aggregate_information(hints)
print(aggregated_output.shape)
