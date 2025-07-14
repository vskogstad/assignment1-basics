import torch
import math
from einops import rearrange, einsum
from torch import nn as nn


class Linear(nn.Module):

    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        std = math.sqrt(2 / (in_features + out_features))
        self.W = nn.Parameter(nn.init.trunc_normal_(tensor=torch.zeros(size= (out_features, in_features)), mean = 0, std = std, a = -3 * std, b = 3 * std))
        

    def forward(self, x: torch.Tensor):
        x = einsum(x, self.W, "... d_in, d_out d_in -> ... d_out")
        return x           


class Embedding(nn.Module):

    def __init__(self, num_embeddings: int, embeddings_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.embedding = nn.Parameter(nn.init.trunc_normal_(tensor=torch.zeros(size= (num_embeddings, embeddings_dim)), mean = 0, std = 1, a = -3, b = 3))
        

    def forward(self, token_ids: torch.Tensor):
        # Pluck out the position for each token_Id

        return self.embedding[token_ids] 

if __name__ == "__main__":
    a = Linear(2, 3)
    print(a.state_dict)
    x = torch.ones((4,3))
    print(a(x))