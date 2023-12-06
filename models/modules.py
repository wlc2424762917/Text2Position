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
                    nn.Linear(channels[i - 1], channels[i]), nn.BatchNorm1d(channels[i]), nn.ReLU()
                )
                for i in range(1, len(channels))
            ]
        )
    else:
        return nn.Sequential(
            *[
                nn.Sequential(nn.Linear(channels[i - 1], channels[i]), nn.ReLU())
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
        padded_indices = np.zeros((batch_size, max_length), np.int32)

        for i, caption_length in enumerate(description_lengths):
            padded_indices[i, :caption_length] = word_indices[i]

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

# class Clip_LanguageEncoder():
#     def __init__(self, clip_version, device):
#         self.device = device
#         self.clip_version = clip_version
#         self.model, self.preprocess = clip.load(clip_version, device=device)
#
#     def __call__(self, descriptions):
#         with torch.no_grad():
#             text_inputs = torch.cat([clip.tokenize(desc) for desc in descriptions]).to(self.device)  # [B, 77]
#             print("text_inputs", text_inputs.device)
#             text_features = self.model.encode_text(text_inputs)  # [B, 512]
#             text_features = F.normalize(text_features, dim=-1)
#             return text_features

class Clip_LanguageEncoder(nn.Module):
    def __init__(self, clip_version):
        super(Clip_LanguageEncoder, self).__init__()
        self.model, _ = clip.load(clip_version)
        self.model.eval()

    def forward(self, descriptions):
        with torch.no_grad():
            text_inputs = torch.cat([clip.tokenize(desc) for desc in descriptions]).to(self.device)
            # text_inputs.to(self.device)# [B, 77]
            text_features = self.model.encode_text(text_inputs)  # [B, 512]
            text_features = F.normalize(text_features, dim=-1)
            # convert to float32, but dont change device
            text_features = text_features.type(torch.FloatTensor).to(self.device)
            return text_features

    @property
    def device(self):
        return next(self.model.parameters()).device
