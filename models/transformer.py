import torch
import torch.nn as nn
import torch.nn.functional as F


class TransfomerLayer(nn.Module):

    def __init__(self, dim, hidden_dim):
        super().__init__()

        # multi-head attention
        self.self_attention = nn.MultiheadAttention(dim, num_heads=8)

        # add & layer normalization
        self.attention_norm = nn.LayerNorm(dim)

        # feed forward
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, dim)

        # add & layer normalization
        self.ff_norm = nn.LayerNorm(dim)


    
    def forward(self, x):
        # Multihead Attention
        attn_output, _ = self.self_attention(x, x, x)
        # Add & Layer Normalization
        x = self.attention_norm(x + attn_output)

        # Feed Forward
        ff_output = self.linear2(F.gelu(self.linear1(x)))
        # Add & Layer Normalization
        x = self.ff_norm(x + ff_output)

        return x
        


class BidirectionalTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # positional embedding
        self.pos_embedding = nn.Embedding(100, 64)

        # transfomer layer
        self.transformer_layers = nn.ModuleList([
            TransfomerLayer(dim=64, hidden_dim=64) for _ in range(6)
        ]) # 6 layers(Can change)
        # read learned codebook
        self.codebook = torch.load('codebook.pt')

    def forward(self, x):
        # positional embedding
        x = x + self.pos_embedding(torch.arange(x.size(1), device=x.device).expand(x.size(0), x.size(1)))
        # transformer
        x = self.transformer(x)
        # read codebook
        x = self.codebook[x.argmax(dim=-1)]
        return x
