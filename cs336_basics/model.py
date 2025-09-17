import math
import random

import torch
import torch.nn.functional as F
from einops import einsum, rearrange
from torch import nn as nn

def norm(x: torch.Tensor):
    return F.rms_norm(x, (x.size(-1),))

def norm2(x: torch.Tensor):
    """Implemenation that replaces F.rms_norm

    class RMSNorm(nn.Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(d_model, dtype=torch.float32, device=device))
        self.d_model = d_model

    def forward(self, x: torch.Tensor):
        # convert from incoming dtype to float32 (If mixed precision training)
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # RMS NormRMS
        rootmeansquared = torch.sqrt((1 / self.d_model) * torch.sum(x**2, dim=-1, keepdim=True) + self.eps)
        x = x * self.weights / rootmeansquared
        # convert back to original dtype
        return x.to(in_dtype)
    
    """
    return RMSNorm(d_model = x.size(-1),)
    

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        set_zero: bool = False,
    ):
        super().__init__()
        std = math.sqrt(2 / (in_features + out_features))
        if set_zero:
            self.W = nn.Parameter(torch.zeros(size=(out_features, in_features), device=device))
        else:
            self.W = nn.Parameter(
                nn.init.trunc_normal_(
                    tensor=torch.zeros(size=(out_features, in_features), device=device),
                    mean=0,
                    std=std,
                    a=-3 * std,
                    b=3 * std,
                )
            )

    def forward(self, x: torch.Tensor):
        x = einsum(x, self.W, "... d_in, d_out d_in -> ... d_out")
        return x


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embeddings_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.embedding = nn.Parameter(
            nn.init.trunc_normal_(
                tensor=torch.zeros(size=(num_embeddings, embeddings_dim), device=device, dtype=dtype),
                mean=0,
                std=1,
                a=-3,
                b=3,
            )
        )

    def forward(self, token_ids: torch.Tensor):
        # Pluck out the position for each token_Id
        return self.embedding[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(d_model, dtype=torch.float32, device=device))
        self.d_model = d_model

    def forward(self, x: torch.Tensor):
        # convert from incoming dtype to float32 (If mixed precision training)
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # RMS NormRMS
        norm_factor = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = x * self.weights * norm_factor
        # convert back to original dtype
        return x.to(in_dtype)


class SILU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x / (1 + torch.exp(-x))
        return x


class Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = 1 / (1 + torch.exp(-x))
        return x


class ReLU2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        torch.pow(torch.max(x, torch.zeros_like(x)))
        return x


class SWIGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        std = math.sqrt(2 / (d_model + d_ff))
        self.w1 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.w2 = Linear(in_features=d_ff, out_features=d_model, device=device, dtype=dtype, set_zero=True)
        self.w3 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.silu = SILU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU(x) = W2(SiLU(xW1) ⊙ xW3)
        x1 = self.w1(x)
        x3 = self.w3(x)

        # cross product (GLU)
        hidden = einsum(self.silu(x1), x3, "b s d_ff, b s d_ff-> b s d_ff")
        # project by to normal dimensions
        x = self.w2(hidden)
        return x


class Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        depth: int,
        max_sequence_length: int | None = None,
        theta: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        glu: bool = True,
        num_kv_groups: int | None = None,
    ):
        super().__init__()
        if num_kv_groups:
            self.mha = GroupedQueryAttention(
                d_model=d_model,
                num_heads=num_heads,
                max_sequence_length=max_sequence_length,
                num_kv_groups=num_kv_groups,
                theta=theta,
                device=device,
                dtype=dtype,
            )
        else:
            self.mha = GatedAttention(
                d_model=d_model,
                num_heads=num_heads,
                max_sequence_length=max_sequence_length,
                theta=theta,
                device=device,
                dtype=dtype,
            )
        self.rmsn1 = RMSNorm(d_model=d_model, eps=torch.finfo(torch.bfloat16).eps, device=device)
        self.scaling = depth**-0.5
        if glu:
            self.ffn = SWIGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
        else:
            self.ffn = SILU()
        self.rmsn2 = RMSNorm(d_model=d_model, eps=torch.finfo(torch.bfloat16).eps, device=device)

    def forward(self, x: torch.Tensor):
        """Pre norm"""
        # Attention with prenorm and lns
        x = x + self.mha(self.scaling * self.rmsn1(x))
        # SILU or SWIGLU FFN with prenorm and lns
        x = x + self.ffn(self.scaling * self.rmsn2(x))
        return x


class PostNormBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_sequence_length: int | None = None,
        theta: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        glu: bool = True,
    ):
        super().__init__()
        self.mha = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_sequence_length=max_sequence_length,
            theta=theta,
            device=device,
            dtype=dtype,
        )
        self.rmsn1 = RMSNorm(d_model=d_model, eps=1e-5, device=device)
        self.ffn = SWIGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
        self.rmsn2 = RMSNorm(d_model=d_model, eps=1e-5, device=device)

    def forward(self, x: torch.Tensor):
        """Post norm"""
        x = self.rmsn1(x + self.mha(x))  # Attention with postorm
        x = self.rmsn2(x + self.ffn(x))  # SWIGLU FFN with postnorm
        return x


class NoLayerNormBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_sequence_length: int | None = None,
        theta: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        glu: bool = True,
    ):
        super().__init__()
        self.mha = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_sequence_length=max_sequence_length,
            theta=theta,
            device=device,
            dtype=dtype,
        )
        self.ffn = SWIGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor):
        """Post norm"""
        x = x + self.mha(x)  #
        x = x + self.ffn(x)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        context_length: int | None = None,
        theta: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        glu: bool = True,
        pre_norm: bool = True,
        layer_norm: bool = True,
    ):
        super().__init__()
        self.embedding = Embedding(num_embeddings=vocab_size, embeddings_dim=d_model, device=device, dtype=dtype)
        block_module = self.get_block_module(
            glu, pre_norm, layer_norm
        )  # For ablations, created separate modules to avoid branching in forward.
        self.layers = nn.Sequential(
            *[
                block_module(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    depth=i + 1,
                    max_sequence_length=context_length,
                    theta=theta,
                    device=device,
                    dtype=dtype,
                    glu=glu,
                )
                for i in range(num_layers)
            ]
        )
        self.rmsn_f = RMSNorm(d_model=d_model, eps=torch.finfo(torch.bfloat16).eps, device=device, dtype=dtype)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size, device=device, dtype=dtype, set_zero=True)
        self.device = device

    def forward(self, x: torch.Tensor):
        x = self.embedding(x)
        x = self.layers(x)
        x = self.rmsn_f(x)
        x = self.lm_head(x)
        return x 

    def get_block_module(self, glu, pre_norm, layer_norm):
        if glu + pre_norm + layer_norm < 2:  # All of these are true by default, we just set one of them false at a time
            raise NotImplementedError("Only setup to do one ablation at a time")

        if not layer_norm:
            return NoLayerNormBlock
        elif not pre_norm:
            return PostNormBlock

        return Block  # If not GLU or if no ablations, we will return the regular block.

    def sample(
        self,
        tokenizer,
        prompt: str | None = None,
        num_samples: int = 1,
        max_tokens: int = 256,
        temp: int = 1,
        top_p: int = 0.5,
    ):
        """
        Sampling using Top-p.

        Top p samples from the n most probable responses which sum is >= top_p.
        Unlike top_k which always samples from the k highest probs. n's range of possibilities is dynamic, while k is fixed.
        """
        with torch.no_grad():
            sequence = tokenizer.encode(prompt) if prompt else tokenizer.encode("\n")
            # end_of_text_token = tokenizer.encode("<|endoftext|>")[0]
            sequence = torch.tensor(sequence, dtype=torch.long, device=self.device).unsqueeze(0).repeat(num_samples, 1)
            # print(f"{sequence.shape = }")

            # turn sequence into torch tensor
            while sequence.shape[-1] < max_tokens:
                out = self(sequence)  # forward pass
                logits = softmax(out, dimension=-1, temp=temp)
                logits = logits[:, -1, :]

                # sort each batch while storing orignial indexes
                sorted, indices = torch.sort(logits, -1, descending=True)
                # print(sorted, indices)

                # torch.cumsum() accumualtes the numbers, find those which values < p
                cum_sum_probs = torch.cumsum(sorted, -1)
                top_probs = cum_sum_probs < top_p  # this is off-by-one to our target
                # print(f"{torch.ones((num_samples, 1)).shape=} | {top_probs[:,:-1].shape}")
                # Add an initial True to our tensors to correct for off-by-one
                top_probs = torch.cat(
                    (torch.ones((num_samples, 1), dtype=bool, device=self.device), top_probs[:, :-1]), dim=-1
                )
                # print(top_probs)

                # mask out probabilities that are not in top_p
                sorted[~top_probs] = 0

                # Sample
                sampled_indices = torch.multinomial(
                    sorted,
                    num_samples=1,
                )

                next_token = torch.gather(indices, dim=-1, index=sampled_indices)

                """p, indices = torch.sort(logits, dim=-1)
                i = 0
                p_accum = 0
                while p_accum < top_p:
                    p_accum += p[i]
                    i += 1
                p_mod = p[:i] / p_accum
                indices_mod = indices[:i]
                next_token = (
                    random.choices(population=indices_mod, weights=p_mod, k=1)[-1].unsqueeze(0).unsqueeze(0)
                )  # adding additional dimensions
                """
                sequence = torch.cat((sequence, next_token), dim=-1)

        for i in range(num_samples):
            print(tokenizer.decode(sequence[i, :].squeeze().tolist()) + "\n")
        return


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_sequence_length: int, device: torch.device | None = None):
        """
        Interleaved RoPE implementation - treats consecutive pairs as complex numbers
        This matches the mathematical approach expected by tests.
        """
        super().__init__()
        self.base = theta
        self.d_k = d_k

        freq = 1.0 / (self.base ** (2 * torch.arange(0, self.d_k / 2.0).float() / (self.d_k))).to(device)
        position = torch.arange(max_sequence_length).to(device)

        # Create position-frequency matrix [max_seq_len, d_k/2]
        pos_freq = torch.einsum("m,f -> mf", position, freq)

        # For interleaved approach: repeat each frequency for consecutive pairs
        # [freq0, freq0, freq1, freq1, freq2, freq2, ...]
        pos_freq_interleaved = torch.zeros(max_sequence_length, d_k, device=device)
        pos_freq_interleaved[:, 0::2] = pos_freq  # Even indices: 0, 2, 4, ...
        pos_freq_interleaved[:, 1::2] = pos_freq  # Odd indices: 1, 3, 5, ...

        # print(f"{d_k=}, {theta=}, {freq=}, {pos_freq=}")

        # Register cos and sin buffers
        self.register_buffer(name="cos", tensor=pos_freq_interleaved.cos(), persistent=False)
        self.register_buffer(name="sin", tensor=pos_freq_interleaved.sin(), persistent=False)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Except for comments, this is basically rewriting Meta's implementation with help from Claude. I had implemeted the transformer-version of Neox-RoPE,
        but that fails the test.

        Interleaved rotation: (-x2, x1, -x4, x3, -x6, x5, ...)
        This treats consecutive pairs as complex numbers: (x1, x2) -> (-x2, x1)

        """
        # Reshape to treat consecutive elements as pairs
        # This line splits the last dimension into pairs, *.shape[:-1] simply keeps the first dimensions as they are
        x_pairs = x.view(*x.shape[:-1], -1, 2)  # [..., d_k/2, 2]
        x1, x2 = x_pairs.unbind(
            dim=-1
        )  # Split along final dimension creating x1 = [1, 3, 5 ...] and x2 = [2, 4, 6 ...]

        # Rotate: (x1, x2) -> (-x2, x1)
        rotated_pairs = torch.stack((-x2, x1), dim=-1)  # [..., d_k/2, 2]

        # Reshape back to original shape
        return rotated_pairs.view(*x.shape)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE with interleaved rotation approach
        """
        # print(f"{x.shape=}")

        # Get cos and sin values for the specified positions
        cos_pos = self.cos[token_positions]  # [seq_len, d_k]
        sin_pos = self.sin[token_positions]  # [seq_len, d_k]

        # print(f"{cos_pos.shape=}")

        # Apply RoPE: x * cos + rotate_half(x) * sin
        x_rope = (x * cos_pos) + (self.rotate_half(x) * sin_pos)

        # print(f"{x_rope.shape=}")
        return x_rope


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_sequence_length: int | None = None,
        theta: float | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.d_k = self.d_v = int(d_model / num_heads)
        # head_size = int(d_model/num_heads)
        self.Wq = Linear(num_heads * self.d_k, d_model, device=device, dtype=dtype, set_zero=True)
        self.Wk = Linear(num_heads * self.d_k, d_model, device=device, dtype=dtype)
        self.Wv = Linear(num_heads * self.d_v, d_model, device=device, dtype=dtype)
        self.Wo = Linear(d_model, num_heads * self.d_v, device=device, dtype=dtype, set_zero=True)
        # self.heads = [Head(head_size=head_size, dim=d_k for _ in range(num_heads)]
        # self.register_buffer(name="tril", tensor=torch.tril(torch.ones((d_model,d_model))))
        if max_sequence_length is None:
            max_sequence_length = (
                d_model  # set it to some slightly large number to pass tests, should be using max_sequence length.
            )
        self.tril = torch.tril(torch.ones((max_sequence_length, max_sequence_length), device=device, dtype=dtype))
        if theta is not None:
            self.rope = RoPE(theta=theta, d_k=self.d_k, max_sequence_length=max_sequence_length, device=device)
        else:
            self.rope = None

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        # We split the embedding dimension into an additional batch dimension (heads)
        Q = rearrange(self.Wq(x), "b s (head d_k) -> b head s d_k", d_k=self.d_k)
        K = rearrange(self.Wk(x), "b s (head d_k) -> b head s d_k", d_k=self.d_k)
        V = rearrange(self.Wv(x), "b s (head d_v) -> b head s d_v", d_v=self.d_v)

        if self.rope != None:  # We are using RoPE
            if token_positions == None:  # create default 0, 1, .... positions if nothing else is supplied
                token_positions = torch.arange(Q.shape[-2])  # Seems brittle
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        # mha = torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=self.tril)
        mha = scaled_dot_product_attention(Q, K, V, mask=self.tril)
        # rearrenge back into original embedding dimension
        mha = rearrange(mha, "b head s d_v -> b s (head d_v)")
        # import sys; sys.exit()
        return self.Wo(mha)


class GatedHeadwiseAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_sequence_length: int | None = None,
        theta: float | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.d_k = self.d_v = int(d_model / num_heads)
        # head_size = int(d_model/num_heads)
        self.Wq = Linear(num_heads * self.d_k, d_model, device=device, dtype=dtype, set_zero=True)
        self.Wk = Linear(num_heads * self.d_k, d_model, device=device, dtype=dtype)
        self.Wv = Linear(num_heads * self.d_v, d_model, device=device, dtype=dtype)
        self.Wo = Linear(d_model, num_heads * self.d_v, device=device, dtype=dtype, set_zero=True)

        self.head_gates = Linear(d_model, num_heads, device=device, dtype=dtype)
        # self.w2 = Linear(in_features=d_ff, out_features=d_model, device=device, dtype=dtype, set_zero=True)
        # self.w3 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.sigmoid = Sigmoid()
        # self.heads = [Head(head_size=head_size, dim=d_k for _ in range(num_heads)]
        # self.register_buffer(name="tril", tensor=torch.tril(torch.ones((d_model,d_model))))
        if max_sequence_length is None:
            max_sequence_length = (
                d_model  # set it to some slightly large number to pass tests, should be using max_sequence length.
            )
        self.tril = torch.tril(torch.ones((max_sequence_length, max_sequence_length), device=device, dtype=dtype))
        if theta is not None:
            self.rope = RoPE(theta=theta, d_k=self.d_k, max_sequence_length=max_sequence_length, device=device)
        else:
            self.rope = None

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        # We split the embedding dimension into an additional batch dimension (heads)
        Q = rearrange(self.Wq(x), "b s (head d_k) -> b head s d_k", d_k=self.d_k)
        K = rearrange(self.Wk(x), "b s (head d_k) -> b head s d_k", d_k=self.d_k)
        V = rearrange(self.Wv(x), "b s (head d_v) -> b head s d_v", d_v=self.d_v)

        if self.rope != None:  # We are using RoPE
            if token_positions == None:  # create default 0, 1, .... positions if nothing else is supplied
                token_positions = torch.arange(Q.shape[-2])  # Seems brittle
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        # mha = torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=self.tril)
        mha = scaled_dot_product_attention(Q, K, V, mask=self.tril)
        # rearrenge back into original embedding dimension
        

        # SwiGLU(x) = W2(SiLU(xW1) ⊙ xW3)
        head_gates = self.sigmoid(self.head_gates(x))
        head_gates = rearrange(head_gates, "b s head -> b head s 1")

        # cross product (GLU)
        hidden = head_gates * mha
        hidden = rearrange(hidden, "b head s d_v -> b s (head d_v)")
        # project by to normal dimensions
        return self.Wo(hidden)




class GatedAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_sequence_length: int | None = None,
        theta: float | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.d_k = self.d_v = int(d_model / num_heads)
        # head_size = int(d_model/num_heads)
        self.Wq = Linear(num_heads * self.d_k, d_model, device=device, dtype=dtype, set_zero=True)
        self.Wk = Linear(num_heads * self.d_k, d_model, device=device, dtype=dtype)
        self.Wv = Linear(num_heads * self.d_v, d_model, device=device, dtype=dtype)
        self.Wo = Linear(d_model, num_heads * self.d_v, device=device, dtype=dtype, set_zero=True)
        self.w1 = Linear(in_features=d_model, out_features=d_model, device=device, dtype=dtype)
        self.q_norm = RMSNorm(d_model=self.d_k, eps=torch.finfo(torch.bfloat16).eps, device=device, dtype=dtype)
        self.k_norm = RMSNorm(d_model=self.d_k, eps=torch.finfo(torch.bfloat16).eps, device=device, dtype=dtype)
        #self.w2 = Linear(in_features=d_ff, out_features=d_model, device=device, dtype=dtype, set_zero=True)
        #self.w3 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.silu = SILU()
        # self.heads = [Head(head_size=head_size, dim=d_k for _ in range(num_heads)]
        # self.register_buffer(name="tril", tensor=torch.tril(torch.ones((d_model,d_model))))
        if max_sequence_length is None:
            max_sequence_length = (
                d_model  # set it to some slightly large number to pass tests, should be using max_sequence length.
            )
        self.tril = torch.tril(torch.ones((max_sequence_length, max_sequence_length), device=device, dtype=dtype))
        if theta is not None:
            self.rope = RoPE(theta=theta, d_k=self.d_k, max_sequence_length=max_sequence_length, device=device)
        else:
            self.rope = None

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        # We split the embedding dimension into an additional batch dimension (heads)
        Q = rearrange(self.Wq(x), "b s (head d_k) -> b head s d_k", d_k=self.d_k)
        K = rearrange(self.Wk(x), "b s (head d_k) -> b head s d_k", d_k=self.d_k)
        V = rearrange(self.Wv(x), "b s (head d_v) -> b head s d_v", d_v=self.d_v)

        if self.rope != None:  # We are using RoPE
            if token_positions == None:  # create default 0, 1, .... positions if nothing else is supplied
                token_positions = torch.arange(Q.shape[-2])  # Seems brittle
            Q, K = self.q_norm(Q), self.k_norm(K) # QK norm. own variant of speedrun implementation from @Grad62304977
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        # mha = torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=self.tril)
        mha = scaled_dot_product_attention(Q, K, V, mask=self.tril)
        # rearrenge back into original embedding dimension
        mha = rearrange(mha, "b head s d_v -> b s (head d_v)")
        

        # SwiGLU(x) = W2(SiLU(xW1) ⊙ xW3)
        x1 = self.w1(x)
        x3 = mha
        # cross product (GLU)
        hidden = self.silu(x1) * x3
        # project by to normal dimensions
        return self.Wo(hidden)




class GroupedQueryAttention(nn.Module):
    """
    K and V are group the attention heads into larger groups.

    Q shape: [batch, num_heads, seq_len, d_k]
    K shape: [batch, num_kv_groups, seq_len, d_k]
    V shape: [batch, num_kv_groups, seq_len, d_v]
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_groups: int,
        max_sequence_length: int | None = None,
        theta: float | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.d_k = self.d_v = int(d_model / num_heads)

        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads / num_kv_groups  # each group consists of group_size number of heads

        # print(f"{type(num_kv_groups * self.d_k)=} | {type(d_model) =}")
        self.Wq = Linear(d_model, num_heads * self.d_k, device=device, dtype=dtype, set_zero=True)
        self.Wk = Linear(d_model, num_kv_groups * self.d_k, device=device, dtype=dtype)
        self.Wv = Linear(d_model, num_kv_groups * self.d_v, device=device, dtype=dtype)
        self.Wo = Linear(num_heads * self.d_v, d_model, device=device, dtype=dtype, set_zero=True)

        # self.register_buffer(name="tril", tensor=torch.tril(torch.ones((d_model,d_model))))
        if max_sequence_length is None:
            max_sequence_length = (
                d_model  # set it to some slightly large number to pass tests, should be using max_sequence length.
            )
        self.tril = torch.tril(torch.ones((max_sequence_length, max_sequence_length), device=device, dtype=dtype))
        if theta is not None:
            self.rope = RoPE(theta=theta, d_k=self.d_k, max_sequence_length=max_sequence_length, device=device)
        else:
            self.rope = None

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        # We split the embedding dimension into an additional batch dimension (heads)
        Q = rearrange(self.Wq(x), "b s (head d_k) -> b head s d_k", d_k=self.d_k)
        K = rearrange(
            self.Wk(x),
            "b s (group d_k) -> b group s d_k",
            d_k=self.d_k,
        )
        V = rearrange(self.Wv(x), "b s (group d_v) -> b group s d_v", d_v=self.d_v)

        if self.rope != None:  # We are using RoPE
            if token_positions == None:  # create default 0, 1, .... positions if nothing else is supplied
                token_positions = torch.arange(Q.shape[-2])  # Seems brittle
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        # mha = torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=self.tril)
        mha = scaled_dot_product_gqa(Q, K, V, mask=self.tril, num_kv_groups=self.num_kv_groups)
        # rearrenge back into original embedding dimension
        mha = rearrange(mha, "b head s d_v -> b s (head d_v)")
        # import sys; sys.exit()

        return self.Wo(mha)


@staticmethod
# @torch.compile(fullgraph=True)
def scaled_dot_product_gqa(Q, K, V, mask, num_kv_groups):
    """
    K and V are group the attention heads into larger groups.

    Q shape: [batch, num_heads, seq_len, d_k]
    K shape: [batch, num_kv_groups, seq_len, d_k]
    V shape: [batch, num_kv_groups, seq_len, d_v]
    """
    d_k = Q.shape[-1]
    seq_len = Q.shape[-2]
    # print(f"{Q.shape=}  {K.shape=} | {V.shape=}")

    # V = rearrange(self.Wv(x), "b s (group d_v) -> b group s d_v", d_v=self.d_v)
    Q_grouped = rearrange(Q, "b (group heads_per_group) sq d_k -> b group heads_per_group sq d_k", group=num_kv_groups)
    K = rearrange(K, "b group sk d_k ->  b group 1 sk d_k")
    V = rearrange(V, "b group sk d_v ->  b group 1 sk d_v")
    # print(f"{Q_grouped.shape=}  {K.shape=} | {V.shape=}")
    attn = einsum(
        Q_grouped, K, "b group heads_per_group sq d_k, b group heads_per_group sk d_k -> b group heads_per_group sq sk"
    ) / math.sqrt(d_k)
    # apply mask if included
    if mask is not None:
        attn = attn.masked_fill(mask[:seq_len, :seq_len] == False, float("-inf"))
    result = einsum(
        softmax(x=attn, dimension=-1),
        V,
        "b group heads_per_group sq sk, b group heads_per_group sk d_v -> b group heads_per_group sq d_v",
    )
    result = rearrange(result, "b group heads_per_group sq d_v -> b (group heads_per_group) sq d_v")
    return result


@torch.compile(fullgraph=True)
def softmax(x: torch.Tensor, dimension: int, temp: int = 1):
    """TODO: Check stability of this might be better with just torch.max(x)"""
    max_x = torch.max(x, dim=dimension, keepdim=True)[0]
    x_mod = (x - max_x) / temp
    result = torch.exp(x_mod) / torch.sum(torch.exp(x_mod), dim=dimension, keepdim=True)
    return result


@torch.compile()
def scaled_dot_product_attention(Q, K, V, mask):
    d_k = Q.shape[-1]
    seq_len = Q.shape[-2]
    # print(f"{Q.shape=}  {K.shape=} | {V.shape=}")
    # Q^T K / sqrt(d_k)
    attn = einsum(Q, K, "b ... sq d_k, b ... sk d_k -> b ... sq sk") / math.sqrt(d_k)
    # apply mask if included
    if mask is not None:
        attn = attn.masked_fill(mask[:seq_len, :seq_len] == False, float("-inf"))
    result = einsum(softmax(x=attn, dimension=-1), V, "b ... sq sk, b ... sk d_v -> b ... sq d_v")

    return result


if __name__ == "__main__":
    a = Linear(2, 3)
    print(a.state_dict)
    x = torch.ones((4, 3))
    print(a(x))
