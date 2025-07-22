import math

import torch
from einops import einsum, rearrange
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

  
class RMSNorm(nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(d_model, dtype=torch.float32))
        self.d_model = d_model


    def forward(self, x: torch.Tensor):
        # convert from incoming dtype to float32 (If mixed precision training)
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # RMS NormRMS
        rootmeansquared =  torch.sqrt((1 / self.d_model) * torch.sum(x**2, dim=-1, keepdim=True) + self.eps)
        x =  x * self.weights / rootmeansquared
        # convert back to original dtype
        return x.to(in_dtype)

class SILU(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x


class SWIGLU(nn.Module):

    def __init__(self, d_model: int, d_ff: int, device: torch.device | None=None, dtype: torch.dtype | None=None):
        super().__init__()
        std = math.sqrt(2 / (d_model + d_ff))
        self.w1 = nn.Parameter(nn.init.trunc_normal_(tensor=torch.zeros(size = (d_ff, d_model), device=device), mean = 0, std = std, a = -3 * std, b = 3 * std))
        self.w2 = nn.Parameter(nn.init.trunc_normal_(tensor=torch.zeros(size = (d_model, d_ff), device=device), mean = 0, std = std, a = -3 * std, b = 3 * std))
        self.w3 = nn.Parameter(nn.init.trunc_normal_(tensor=torch.zeros(size = (d_ff, d_model), device=device), mean = 0, std = std, a = -3 * std, b = 3 * std))
        self.silu = SILU()



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU(x) = W2(SiLU(xW1) ⊙ xW3)
        x1 = einsum(self.w1, x, "d_ff d_model, b s d_model -> b s d_ff")
        x3 = einsum(self.w3, x, "d_ff d_model, b s d_model -> b s d_ff" )

        #cross product (GLU)
        hidden = einsum(self.silu(x1), x3, "b s d_ff, b s d_ff-> b s d_ff") 
        # project by to normal dimensions
        x = einsum(self.w2, hidden, "d_model d_ff, b s d_ff -> b s d_model")
        return x
    
class RoPE(nn.Module):

    def __init__(self, theta: float, d_k: int, max_sequence_length: int, device: torch.device | None = None):
        """
        Pseudocode from original RoPE paper:

        sinusoidal_pos.shape = [1, seq_len, hidden_size] # Sinusoidal position embeddings
        qw.shape = [batch_size, seq_len, num_heads, hidden_size]  # query hiddens
        kw.shape = [batch_size, seq_len, num_heads, hidden_size]  # key hiddens

        cos_pos = repeat_elements(sinusoidal_pos[..., None, 1::2], rep=2, axis=-1)
        sin_pos = repeat_elements(sinusoidal_pos[..., None, ::2], rep=2, axis=-1)
        qw2 = stack([-qw[..., 1::2], qw[..., ::2]], 4)
        qw2 = reshape(qw2, shape(qw))
        qw = qw * cos_pos + qw2 * sin_pos
        kw2 = K.stack([-kw[..., 1::2], kw[..., ::2]], 4)
        kw2 = K.reshape(kw2, K.shape(kw))
        kw = kw * cos_pos + kw2 * sin_pos

        # Attention
        a = tf.einsum('bjhd,bkhd->bhjk', qw, kw)"""
        self.theta = theta
        self.d_k = d_k

        # Implement an R-matrix that can be reused for various calculations of RoPE.
        
        self.register_buffer(name="R_mat", 
                             tensor=torch.cos(max_sequence_length), 
                             persistent=False)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        
        return x
'''
class Head(nn.Module):

    def __init__(self, head_size, dim):
        super().__init__()


    def forward(x: torch.tensor):
        return scaled_dot_product_attention(self.Q, self.K, self.V, mask="tril")
'''

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int, dtype: torch.dtype | None=None, device: torch.device | None=None):
        super().__init__()
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.d_k = self.d_v = int(d_model/num_heads)
        #head_size = int(d_model/num_heads)
        self.Wq = Linear(num_heads * self.d_k, d_model)
        self.Wk = Linear(num_heads * self.d_k, d_model)
        self.Wv = Linear(num_heads * self.d_v, d_model) 
        self.Wo = Linear(d_model, num_heads * self.d_v) 
        #self.heads = [Head(head_size=head_size, dim=d_k for _ in range(num_heads)]
        #self.register_buffer(name="tril", tensor=torch.tril(torch.ones((d_model,d_model))))
        self.tril =torch.tril(torch.ones((d_model,d_model)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mha = []
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        for i in range(self.num_heads):
            start = i * self.d_k
            end = (i + 1) * self.d_k
            
            Qi, Ki, Vi = Q[:,:,start:end], K[:,:,start:end], V[:,:,start:end]
            print(Qi.shape)
            # import sys; sys.exit()
            mha.append(scaled_dot_product_attention(Qi, Ki, Vi, mask=self.tril))
            
        mha = torch.cat(tuple(mha), -1)
        # mha = scaled_dot_product_attention(self.Wq(x), self.Wk(x), self.Wv(x), mask=self.tril)
        return self.Wo(mha)


@staticmethod
def softmax(x: torch.Tensor, dimension: int):
    max_x = torch.max(x[dimension])
    result = torch.exp(x-max_x) / torch.sum(torch.exp(x-max_x), dim=dimension, keepdim=True)
    return result

@staticmethod
def scaled_dot_product_attention(Q, K, V, mask):
    d_k = Q.shape[-1]
    seq_len = Q.shape[-2]
    # print(f"{Q.shape=}  {K.shape=} | {V.shape=}")
    # Q^T K / sqrt(d_k)
    attn = einsum(Q, K, "b ... sq d_k, b ... sk d_k -> b ... sq sk") / math.sqrt(d_k)
    # apply mask if included
    if mask is not None:
        attn = attn.masked_fill(mask[:seq_len,:seq_len]==False, float("-inf"))
    result = einsum(softmax(x=attn, dimension=-1), V, "b ... sq sk, b ... sk d_v -> b ... sq d_v")
    return result

if __name__ == "__main__":
    a = Linear(2, 3)
    print(a.state_dict)
    x = torch.ones((4,3))
    print(a(x))