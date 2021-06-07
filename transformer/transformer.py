import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

from torch import Tensor


def position_encoding(
    seq_len: int, dim_model: int, device: torch.device = torch.device("cpu"),
) -> Tensor:
    
    pos = torch.arange(seq_len, dtype=torch.float, device=device).reshape(1, -1, 1)
    dim = torch.arange(dim_model, dtype=torch.float, device=device).reshape(1, 1, -1)
    phase = pos / 1e4 ** (dim // dim_model)

    return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))


def create_look_ahead_mask(seq):
	"""
	A function to create a binrary mask for attention which prohibts 
	the lookup for the unseen tokens, while predicting the next token of the sequene
	"""

	mask_size = seq.size()[0]
	mask  = np.ones((mask_size, mask_size))
	mask -= np.tril(mask)

	return torch.stack([Tensor(mask) * seq.size(0)], dim=0)

def create_pad_mask(seq, pad_token_id):

	mask = (seq == pad_token_id).cpu().long()
	return torch.stack([mask] * seq.size(1), dim=1) 


def scaled_dot_product_attention(query, key, value, mask=None) -> Tensor:

    temp = query.bmm(key.transpose(1, 2))
    scale = query.size(-1) ** 0.5

    scaled_attn_logits = temp / scale
    if mask is not None:
    	scaled_attn_logits += (mask.to(scaled_attn_logits.device) * (-1e+9))

    softmax = F.softmax(scaled_attn_logits, dim=-1)
    return softmax.bmm(value)


def feed_forward(dim_input: int = 512, dim_feedforward: int = 2048) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dim_input, dim_feedforward),
        nn.ReLU(),
        nn.Linear(dim_feedforward, dim_input),
    )


class AttentionHead(nn.Module):
    def __init__(self, dim_in: int, dim_k: int, dim_v: int):

        super().__init__()
        self.q = nn.Linear(dim_in, dim_k)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_v)

    def forward(self, query, key, value, mask=None) -> Tensor:
        return scaled_dot_product_attention(self.q(query), self.k(key), self.v(value), mask)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_k: int, dim_v: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [AttentionHead(dim_in, dim_k, dim_v) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_v, dim_in)

    def forward(self, query, key, value, mask=None) -> Tensor:
        return self.linear(
            torch.cat([h(query, key, value, mask) for h in self.heads], dim=-1)
        )


class Residual(nn.Module):
    def __init__(self, sublayer: nn.Module, dimension: int, dropout: float = 0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *tensors: Tensor) -> Tensor:
        return self.norm(tensors[0] + self.dropout(self.sublayer(*tensors)))


class TransformerLayer(nn.Module):
    def __init__(
        self, 
        dim_model: int = 512, 
        num_heads: int = 6, 
        dim_feedforward: int = 2048, 
        dropout: float = 0.1, 
    ):
        super().__init__()
        dim_k = dim_v = dim_model // num_heads
        self.attention = Residual(
            MultiHeadAttention(num_heads, dim_model, dim_k, dim_v),
            dimension=dim_model,
            dropout=dropout,
        )
        self.feed_forward = Residual(
            feed_forward(dim_model, dim_feedforward),
            dimension=dim_model,
            dropout=dropout,
        )

    def forward(self, src: Tensor, mask=None):
        src = self.attention(src, src, src, mask)
        return self.feed_forward(src)


class Transformer(nn.Module):
    def __init__(
        self, 
        img_tokens,
        text_tokens,
        seq_len_text,
        seq_len_image,
        num_layers: int = 6,
        dim_model: int = 512, 
        num_heads: int = 8, 
        dim_feedforward: int = 2048, 
        dropout: float = 0.1, 
    ):
        super().__init__()

        self.seq_len_image = seq_len_image
        self.seq_len_text  = seq_len_text

        total_tokens = text_tokens + img_tokens

        self.embedding = nn.Embedding(total_tokens, dim_model)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(dim_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        self.output = Linear(dim_model, total_tokens)
        
    def forward(self, src, mask=None):

        src = self.embedding(src)
        seq_len, dimension = src.size(1), src.size(2)
        
        # add separate pos.encoding for image and text
        src[:text_tokens] += position_encoding(seq_len[:text_tokens], dimension).to(src.device)
        src[text_tokens:] += position_encoding(seq_len[text_tokens:], dimension).to(src.device)

        for layer in self.layers: src = layer(src, mask)

        logits = self.output(src)
        
        return logits