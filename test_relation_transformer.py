import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationMultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(RelationMultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.relations = nn.Linear(self.head_dim, self.head_dim, bias=False)

        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
        self.fc_fuse = nn.Linear(2 * self.head_dim, self.head_dim)

    def forward(self, values, keys, query, mask, relation):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        relation = relation.reshape(N, query_len, key_len, self.heads, self.head_dim)
        # get q k v relation
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        relation = self.relations(relation)
        relation = relation.permute(0, 3, 1, 2, 4)

        # 计算 Q 和 K 的点积，并将 relation 的嵌入加入到乘积中
        # einsum 路径 'nqhd,nkhd,nhqkd->nhqk' 表示对于每个批次 (n) 和每个头 (h)：
        # - Q (nqhd) 和 K (nkhd) 的点积
        # - 然后与 relation (nhqkd) 相乘
        # energy = torch.einsum('nqhd,nkhd,nhqkd->nhqk', queries, keys, relation)

        # energy = torch.einsum('nqhd,nkhd->nhqkd', queries, keys)
        # energy = torch.einsum('nhqkd,nhqkd->nhqk', energy, relation)
        # - Q (nqhd) 和 K (nkhd) 的点积
        energy = torch.einsum('nqhd,nkhd->nhqkd', queries, keys)
        # energy = torch.einsum('nhqkd,nhqkd->nhqk', energy, relation)
        # concat relation and energy
        print(energy.shape)
        energy = torch.cat((energy, relation), dim=-1)
        print(energy.shape)
        # 通过一个mlp
        energy = self.fc_fuse(energy)
        print(energy.shape)
        # 将d维sum到一起
        energy = energy.sum(dim=-1, keepdim=True)

        if mask is not None:
            energy = energy.masked_fill(mask == 1, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        print(out.shape)
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
        mask = torch.ones(len(batch.unique()), max_len, max_len)

        # 对每个批次添加填充embedding, relation掩码, (B, max_len, D), (B, max_len, max_len, embed_dim)
        for i, b in enumerate(batch.unique()):
            # embedding padding
            batch_embeddings = embeddings[batch == b]
            padded_embeddings[i, :batch_embeddings.size(0)] = batch_embeddings
            mask[i, :batch_embeddings.size(0), :batch_embeddings.size(0)] = 0  # 0表示有效值，1表示无效值
            # relation embedding padding
            relation_len = relation[i].shape[0]
            assert relation_len == sum(batch == b)
            padded_relation[i, :relation_len, :relation_len] = relation[i]

        # 调整形状以适应多头注意力模块
        padded_embeddings = padded_embeddings
        # 重塑掩码以匹配多头注意力的头数

        # mask = mask.repeat(self.num_heads, 1, 1)  # 形状变为 [num_heads * B, max_len, max_len]
        mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)  # 形状变为 [B, num_heads, max_len, max_len]

        # 应用多头自注意力机制
        attn_output = self.multihead_attn(padded_embeddings.to(self.device), padded_embeddings.to(self.device), padded_embeddings.to(self.device), mask.to(self.device), padded_relation.to(self.device))

        # 应用最大池化
        attn_output = F.max_pool1d(attn_output.transpose(1, 2), attn_output.size(1)).squeeze(-1)

        return attn_output

    @property
    def device(self):
        return next(self.multihead_attn.parameters()).device


def test_relation_multihead_self_attention_with_mask():
    # Define parameters
    N = 10  # Total number of instances
    embed_dim = 4  # Embedding dimension
    num_heads = 2  # Number of attention heads
    num_batches = 2  # Number of batches
    batch_sizes = [6, 4]  # Number of instances in each batch

    # Create the RelationMultiHeadSelfAttention_withMask instance
    model = MaxPoolRelationMultiHeadSelfAttention(embed_dim, num_heads)

    # Create dummy embeddings
    embeddings = torch.rand(N, embed_dim)

    # Create dummy batch assignments
    batch = torch.cat([torch.full((size,), i) for i, size in enumerate(batch_sizes)])

    # Create dummy relation matrices for each batch
    relation = [torch.rand(size, size, embed_dim) for size in batch_sizes]

    # Run the model
    output = model(embeddings, batch, relation)

    # Print output shape
    print("Output shape:", output.shape)

# Run the test
test_relation_multihead_self_attention_with_mask()

# Example usage of RelationMultiHeadSelfAttention
# embed_size = 256
# heads = 8
# attention = RelationMultiHeadSelfAttention(embed_size, heads)
# x = torch.rand((5, 60, embed_size))  # Example input (batch size, sequence length, embedding size)
# mask = None  # Example mask, can be used for masking out padding tokens in sequences
# relation = torch.rand((5, 60, 60, embed_size))  # Example relation
# output = attention(x, x, x, mask, relation)  # Self attention
# print(output.shape)
