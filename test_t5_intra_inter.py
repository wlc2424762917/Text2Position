import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5EncoderModel, T5Model


class TransformerWithMaxPool(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward):
        super(TransformerWithMaxPool, self).__init__()
        # 创建 Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)  # num_layers stands for the number of sub-encoder-layers in the encoder
        # self.transformer_encoder_1 = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # MaxPooling 层
        self.maxpool = nn.AdaptiveMaxPool1d(1)  # output_size=1

    def forward(self, src):
        # Transformer 编码器处理
        print(f"src.shape: {src.shape}")
        transformer_output = self.transformer_encoder(src)  # [N, B, D]
        # 转换维度以适应 MaxPooling 层
        transformer_output = transformer_output.permute(1, 2, 0)  # [B, D, N]
        # 应用 MaxPooling
        pooled_output = self.maxpool(transformer_output)  # [B, D, 1]
        # 再次转换维度
        pooled_output = pooled_output.permute(2, 0, 1)  # [1, B, D]
        return pooled_output.squeeze(0)  # [B, D]

class T5_LanguageEncoder_TransformerFuser(nn.Module):
    def __init__(self, T5_model_path='/home/wanglichao/t5-small', T5_model_freeze=True):
        super(T5_LanguageEncoder_TransformerFuser, self).__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(T5_model_path)
        self.model = T5EncoderModel.from_pretrained(T5_model_path)

        if T5_model_freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self.transformerD = TransformerWithMaxPool(d_model=512, nhead=8, num_layers=1, dim_feedforward=2048)
        self.transformerN = TransformerWithMaxPool(d_model=512, nhead=8, num_layers=1, dim_feedforward=2048)

    def forward(self, descriptions):
        # all_sentences = [sentence.strip() for description in descriptions for sentence in description.split('.')
        #                  if sentence.strip()]
        all_sentences = []
        max_token_len = 0
        for description in descriptions:
            description = description.replace('.', ' . ')
            sentences = [sentence.strip() for sentence in description.split('.') if sentence.strip()]
            all_sentences.extend(sentences)
            token_lens = [len(self.tokenizer.encode(sentence, add_special_tokens=False)) for sentence in sentences]
            max_token_len = max(max_token_len, max(token_lens, default=0))

        # 编码所有句子
        all_text_inputs = torch.cat([self.tokenizer(sentence,
                                                    return_tensors='pt',
                                                    padding='max_length',
                                                    truncation=True,
                                                    max_length=max_token_len).input_ids for sentence in all_sentences])

        # 获取注意力掩码
        attention_masks = torch.cat([self.tokenizer(sentence,
                                                    return_tensors='pt',
                                                    padding='max_length',
                                                    truncation=True,
                                                    max_length=max_token_len).attention_mask for sentence in all_sentences])

        # 使用模型获取特征
        all_text_features = self.model(input_ids=all_text_inputs, attention_mask=attention_masks).last_hidden_state
        print(all_text_features.shape)
        description_features = []
        start_index = 0
        for description in descriptions:
            num_sentences = 6  # 假设每个描述有6个句子
            end_index = start_index + num_sentences
            description_feature = all_text_features[start_index:end_index]
            description_features.append(description_feature)
            start_index = end_index

        # concatenate the description features not stack
        description_features = torch.stack(description_features, dim=0)  # shape [B, N, D, 512]
        print(description_features.shape)
        # D维度上的Transformer和MaxPooling
        description_features_d = [self.transformerD(description_feature) for description_feature in
                                  description_features.transpose(1, 2)]
        description_features_d = torch.stack(description_features_d, dim=0) # shape [B, N, 512]

        # N维度上的Transformer和MaxPooling
        print(description_features_d.shape)
        description_features_n = description_features_d.permute(1, 0, 2)  # shape [N, B, 512]
        description_features_n = self.transformerN(description_features_n)  # shape [B, 512]

        return description_features_n

def test_T5_LanguageEncoder_TransformerFuser():
    # Create a mock dataset of descriptions
    mock_descriptions = [
        "This is the first description the the as. This is the first description. This is the first description. This is the first description. This is the first description. This is the first description. ",
        "This is the first description. This is the first description. This is the first description. This is the first description. This is the first description. This is the first description. "
        # Add more descriptions as needed
    ]

    # Initialize the T5 Language Encoder Transformer Fuser
    model = T5_LanguageEncoder_TransformerFuser(T5_model_path='/home/wanglichao/t5-small', T5_model_freeze=True)

    # Convert descriptions to tensor
    descriptions_tensor = model.forward(mock_descriptions)

    # Check if the output shape is as expected
    print(descriptions_tensor.shape)
    print(descriptions_tensor.max(), descriptions_tensor.min())

# Run the test
test_T5_LanguageEncoder_TransformerFuser()
