from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import clip

# CARE: This has a trailing ReLU!!
def get_mlp(channels: List[int], add_batchnorm: bool = True) -> nn.Sequential:
    """Construct and MLP for use in other models.

    Args:
        channels (List[int]): List of number of channels in each layer.
        add_batchnorm (bool, optional): Whether to add BatchNorm after each layer. Defaults to True.

    Returns:
        nn.Sequential: Output MLP
    """
    if add_batchnorm:
        return nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(channels[i - 1], channels[i]), nn.BatchNorm1d(channels[i]), nn.GELU()
                )
                for i in range(1, len(channels))
            ]
        )
    else:
        return nn.Sequential(
            *[
                nn.Sequential(nn.Linear(channels[i - 1], channels[i]), nn.GELU())
                for i in range(1, len(channels))
            ]
        )


class LanguageEncoder(torch.nn.Module):
    def __init__(self, known_words, embedding_dim, bi_dir, num_layers=1):
        """Language encoder to encode a set of hints"""
        super(LanguageEncoder, self).__init__()

        self.known_words = {c: (i + 1) for i, c in enumerate(known_words)}
        self.known_words["<unk>"] = 0
        self.word_embedding = nn.Embedding(len(self.known_words), embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            bidirectional=bi_dir,
            num_layers=num_layers,
        )

    """
    Encodes descriptions as batch [d1, d2, d3, ..., d_B] with d_i a string. Strings can be of different sizes.
    """

    def forward(self, descriptions):
        word_indices = [
            [
                self.known_words.get(word, 0)
                for word in description.replace(".", "").replace(",", "").lower().split()
            ]
            for description in descriptions
        ]
        description_lengths = [len(w) for w in word_indices]
        batch_size, max_length = len(word_indices), max(description_lengths)
        padded_indices = np.zeros((batch_size, max_length), np.int32)  # [B, maxLen]

        for i, caption_length in enumerate(description_lengths):
            padded_indices[i, :caption_length] = word_indices[i]  # [B, maxLen] fill with indices

        padded_indices = torch.from_numpy(padded_indices)
        padded_indices = padded_indices.to(self.device)  # Possibly move to cuda

        embedded_words = self.word_embedding(padded_indices)
        description_inputs = nn.utils.rnn.pack_padded_sequence(
            embedded_words,
            torch.tensor(description_lengths),
            batch_first=True,
            enforce_sorted=False,
        )

        d = 2 * self.lstm.num_layers if self.lstm.bidirectional else 1 * self.lstm.num_layers
        h = torch.zeros(d, batch_size, self.word_embedding.embedding_dim).to(self.device)
        c = torch.zeros(d, batch_size, self.word_embedding.embedding_dim).to(self.device)

        _, (h, c) = self.lstm(description_inputs, (h, c))
        description_encodings = torch.mean(h, dim=0)  # [B, DIM] TODO: cat even better?

        return description_encodings

    @property
    def device(self):
        return next(self.lstm.parameters()).device


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


class Clip_LanguageEncoder(nn.Module):
    def __init__(self, clip_version):
        super(Clip_LanguageEncoder, self).__init__()
        self.model, _ = clip.load(clip_version)
        self.model.eval()

    def forward(self, descriptions, freeze=True):
        if freeze:
            with torch.no_grad():
                text_inputs = torch.cat([clip.tokenize(desc) for desc in descriptions]).to(self.device)
                # text_inputs.to(self.device)# [B, 77]
                text_features = self.model.encode_text(text_inputs)  # [B, 512]
                # text_features = F.normalize(text_features, dim=-1)
                # convert to float32, but dont change device
                text_features = text_features.type(torch.FloatTensor).to(self.device)
                return text_features
        else:
            text_inputs = torch.cat([clip.tokenize(desc) for desc in descriptions]).to(self.device)
            text_features = self.model.encode_text(text_inputs)  # [B, 512]
            text_features = text_features.type(torch.FloatTensor).to(self.device)
            return text_features

    @property
    def device(self):
        return next(self.model.parameters()).device


class TransformerWithMaxPool(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward):
        super(TransformerWithMaxPool, self).__init__()
        # 创建 Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # self.transformer_encoder_1 = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # MaxPooling 层
        self.maxpool = nn.AdaptiveMaxPool1d(1)

    def forward(self, src):
        # Transformer 编码器处理
        transformer_output = self.transformer_encoder(src)
        # transformer_output = self.transformer_encoder_1(transformer_output)
        # 转换维度以适应 MaxPooling 层
        transformer_output = transformer_output.permute(1, 2, 0)
        # 应用 MaxPooling
        pooled_output = self.maxpool(transformer_output)
        # 再次转换维度
        pooled_output = pooled_output.permute(2, 0, 1)
        return pooled_output.squeeze(0)


class Clip_LanguageEncoder_TransformerFuser(nn.Module):
    def __init__(self, clip_version, clip_text_freeze=True):
        super(Clip_LanguageEncoder_TransformerFuser, self).__init__()
        self.model, _ = clip.load(clip_version)
        self.clip_text_freeze = clip_text_freeze
        self.model.float()
        if clip_text_freeze:
            for name, param in self.model.named_parameters():
                param.requires_grad_(False)
                # print(name, param.requires_grad)
        self.adapter = Adapter(512, 4)
        self.ratio = 0.2
        self.transformerFuser = TransformerWithMaxPool(d_model=512, nhead=8, num_layers=6, dim_feedforward=2048)

    def forward(self, descriptions):
        all_sentences = [sentence.strip() for description in descriptions for sentence in description.split('.')
                         if sentence.strip()]
        # Tokenize all sentences in one go
        all_text_inputs = torch.cat(
            [clip.tokenize(sentence).to(self.device) for sentence in all_sentences])  # [B*N, 77]
        # Encode all sentences in one go
        all_text_features = self.model.encode_text(all_text_inputs)  # [B*N, 512]
        x = self.adapter(all_text_features)
        all_text_features = self.ratio * x + (1 - self.ratio) * all_text_features

        description_features = []
        start_index = 0
        for description in descriptions:
            num_sentences = 6
            end_index = start_index + num_sentences
            description_feature = all_text_features[start_index:end_index]
            description_features.append(description_feature)
            start_index = end_index
        # Stack all description features
        description_features = torch.stack(description_features, dim=0)  # [B, N, 512]
        description_features = F.normalize(description_features, dim=-1)
        # convert to [N, B, 512]
        description_features = description_features.transpose(0, 1)
        description_features = description_features.type(torch.FloatTensor).to(self.device)
        # inter transformer fusion
        fused_description_features = self.transformerFuser(description_features)
        # normalize
        # fused_description_features = F.normalize(fused_description_features, dim=-1)
        return fused_description_features

    @property
    def device(self):
        return next(self.model.parameters()).device


class MaxPoolMultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MaxPoolMultiHeadSelfAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.num_heads = num_heads

    def forward(self, embeddings, batch):
        # 计算每个批次的最大长度
        max_len = max([sum(batch == i) for i in batch.unique()])

        # 初始化填充后的嵌入和掩码
        padded_embeddings = torch.zeros(len(batch.unique()), max_len, embeddings.size(-1))
        mask = torch.ones(len(batch.unique()), max_len, max_len)

        # 对每个批次添加填充并创建掩码, (B, max_len, D)
        for i, b in enumerate(batch.unique()):
            # print(batch == b)
            # print(embeddings[batch == b].shape)
            batch_embeddings = embeddings[batch == b]
            padded_embeddings[i, :batch_embeddings.size(0)] = batch_embeddings
            mask[i, :batch_embeddings.size(0), :batch_embeddings.size(0)] = 0

        # 调整形状以适应多头注意力模块
        padded_embeddings = padded_embeddings.transpose(0, 1)  # (max_len, B, D)

        # 重塑掩码以匹配多头注意力的头数
        mask = mask.repeat(self.num_heads, 1, 1)  # 形状变为 [num_heads * B, max_len, max_len]

        # 应用多头自注意力机制
        attn_output, _ = self.multihead_attn(padded_embeddings.to(self.device), padded_embeddings.to(self.device), padded_embeddings.to(self.device), attn_mask=mask.to(self.device))
        attn_output = attn_output.transpose(0, 1)  # (B, max_len, D)

        # 应用最大池化
        maxpooled_output = F.max_pool1d(attn_output.transpose(1, 2), kernel_size=attn_output.size(1))  # (B, D, 1)

        return maxpooled_output.squeeze()

    @property
    def device(self):
        return next(self.multihead_attn.parameters()).device


class RelationMultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(RelationMultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"
        # 初始化CLS token的嵌入
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_size))
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.relations = nn.Linear(self.head_dim, self.head_dim, bias=False)

        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask, relation):
        N = query.shape[0]
        # 使用CLS token嵌入
        cls_tokens = self.cls_token.repeat(N, 1, 1)  # 复制CLS token以匹配批次大小
        values = torch.cat([cls_tokens, values], dim=1)
        keys = torch.cat([cls_tokens, keys], dim=1)
        query = torch.cat([cls_tokens, query], dim=1)

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        relation = relation.reshape(N, query_len - 1, key_len - 1, self.heads, self.head_dim)

        # Pool the relation matrix along rows and columns using max pooling
        relation_row_pool, _ = relation.max(dim=-3, keepdim=False)
        relation_col_pool, _ = relation.max(dim=-4, keepdim=False)

        # Get q, k, v from linear transformations
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Add the pooled relation matrix to q and k,
        # 分离第一个元素和剩余部分
        queries_first, queries_rest = queries[:, :1, :, :], queries[:, 1:, :, :]
        keys_first, keys_rest = keys[:, :1, :, :], keys[:, 1:, :, :]

        # 对剩余部分执行加法操作
        queries_rest_with_relation = queries_rest + relation_row_pool
        keys_rest_with_relation = keys_rest + relation_col_pool

        # 重组queries和keys
        queries_with_relation = torch.cat([queries_first, queries_rest_with_relation], dim=1)
        keys_with_relation = torch.cat([keys_first, keys_rest_with_relation], dim=1)

        # Calculate the energy (qk^T)
        energy = torch.einsum('nqhd,nkhd->nhqk', queries_with_relation, keys_with_relation)

        if mask is not None:
            energy = energy.masked_fill(mask == 1, float("-1e20"))

        # Softmax
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        # Perform attention-based weighted sum
        out = torch.einsum("nhqk,nqhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out


class MaxPoolRelationMultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MaxPoolRelationMultiHeadSelfAttention, self).__init__()
        self.multihead_attn = RelationMultiHeadSelfAttention(embed_dim, num_heads)
        self.num_heads = num_heads
        self.embed_dim = embed_dim

    def forward(self, embeddings, batch, relation):
        # relation is [batch_size, num_obj, num_obj, embed_dim]
        # 计算每个批次的最大长度
        max_len = max([sum(batch == i) for i in batch.unique()])

        # 初始化填充后的嵌入和掩码
        padded_embeddings = torch.zeros(len(batch.unique()), max_len, embeddings.size(-1))
        padded_relation = torch.zeros(len(batch.unique()), max_len, max_len, self.embed_dim)
        mask = torch.ones(len(batch.unique()), max_len+1, max_len+1) # consider the cls token

        # 对每个批次添加填充embedding, relation掩码, (B, max_len, D), (B, max_len, max_len, embed_dim)
        for i, b in enumerate(batch.unique()):
            # embedding padding
            batch_embeddings = embeddings[batch == b]
            padded_embeddings[i, :batch_embeddings.size(0)] = batch_embeddings
            mask[i, :batch_embeddings.size(0)+1, :batch_embeddings.size(0)+1] = 0  # 0表示有效值，1表示无效值
            # relation embedding padding
            relation_len = relation[i].shape[0]
            # assert relation_len == sum(batch == b)
            # padded_relation[i, :relation_len, :relation_len] = relation[i]
            assert relation_len == max_len
            padded_relation[i] = relation[i]

        # 调整形状以适应多头注意力模块
        padded_embeddings = padded_embeddings
        # 重塑掩码以匹配多头注意力的头数

        # mask = mask.repeat(self.num_heads, 1, 1)  # 形状变为 [num_heads * B, max_len, max_len]
        mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)  # 形状变为 [B, num_heads, max_len, max_len]

        # 应用多头自注意力机制
        attn_output = self.multihead_attn(padded_embeddings.to(self.device), padded_embeddings.to(self.device), padded_embeddings.to(self.device), mask.to(self.device), padded_relation.to(self.device))
        # attn_output [B, max_len, D]

        # 取global token
        attn_output = attn_output[:, 0, :]  # [B, D]

        return attn_output

    @property
    def device(self):
        return next(self.multihead_attn.parameters()).device
