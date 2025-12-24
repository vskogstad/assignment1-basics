import math

import torch
import triton
import triton.language as tl
from einops import einsum, rearrange, reduce

"""@triton.autotune(
    configs = [
        triton.Config(
            {"Q_TILE_SIZE": Q_TILE_SIZE, "K_TILE_SIZE":K_TILE_SIZE},
            num_warps=num_warps,
        )
        for Q_TILE_SIZE in [32, 64]
        for K_TILE_SIZE in [32, 64]
        for num_warps in [2, 4]
    ],
    key = ["N_QUERIES", "N_KEYS", "D"]
)"""
@triton.jit
def flash_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,
    N_QUERIES,
    N_KEYS,
    scale,
    is_causal: tl.constexpr,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    # Set up K_block_ptr, V_block_ptr, O_block_ptr, L_block_ptr similarly
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )



    m_curr = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)
    l_j = tl.full((Q_TILE_SIZE,), 1, dtype=tl.float32)
    o_j = tl.full((Q_TILE_SIZE, D), 1, dtype=tl.float32)
    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    last_query_pos = (query_tile_index + 1) * Q_TILE_SIZE - 1
    num_k_tiles = last_query_pos // K_TILE_SIZE + 1 if is_causal else N_KEYS // K_TILE_SIZE


    if is_causal:
        start_diag_tile = query_tile_index * Q_TILE_SIZE // K_TILE_SIZE

    else: 
        start_diag_tile = num_k_tiles

    
    row_idx = tl.arange(0, Q_TILE_SIZE) + query_tile_index * Q_TILE_SIZE
    K_base = tl.arange(0, K_TILE_SIZE)

    # regular loop
    for j in range(start_diag_tile):
        k_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        # tl.device_print("k_j", k_j)
        s_ij = tl.dot(q, tl.trans(k_j)) * scale

        m_j = tl.max(s_ij, axis=-1)
        m_j = tl.maximum(m_curr, m_j)  # find max values for all tiles up to and including j
        p_ij = tl.exp(s_ij - m_j[:, None])  # running_numerator (this is NOT a normalized complete P_ij tile)  Bq x Bk
        l_j = tl.exp(m_curr - m_j) * l_j + tl.sum(p_ij, axis=-1)
        # running_denominator: add current sum, scale previous sum by e(m_{j-i} - m_{j})
        o_j = tl.exp(m_curr - m_j)[:, None] * o_j + tl.dot(
            p_ij.to(v_j.dtype), v_j
        )  # running output. Bq, Bq @ Bq, d -> Bq, d
        m_curr = m_j  # update m j-1

        # advance k, v - pointers
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    # Diagonals in separate loop:
    for j_diag in range(start_diag_tile, num_k_tiles):
        k_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        # tl.device_print("k_j", k_j)
        s_ij = tl.dot(q, tl.trans(k_j)) * scale

        # Masking
        col_idx = K_base + j_diag * K_TILE_SIZE
        causal_mask = row_idx[:, None] >= col_idx[None, :]
        s_ij = tl.where(causal_mask, s_ij, float("-inf"))

        m_j = tl.max(s_ij, axis=-1)
        m_j = tl.maximum(m_curr, m_j)  # find max values for all tiles up to and including j
        p_ij = tl.exp(s_ij - m_j[:, None])  # running_numerator (this is NOT a normalized complete P_ij tile)  Bq x Bk
        l_j = tl.exp(m_curr - m_j) * l_j + tl.sum(p_ij, axis=-1)
        # running_denominator: add current sum, scale previous sum by e(m_{j-i} - m_{j})
        o_j = tl.exp(m_curr - m_j)[:, None] * o_j + tl.dot(
            p_ij.to(v_j.dtype), v_j
        )  # running output. Bq, Bq @ Bq, d -> Bq, d
        m_curr = m_j  # update m j-1

        # advance k, v - pointers
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))


    o_j = (1 / l_j)[:, None] * o_j
    l_j = m_curr + tl.log(l_j)

    tl.store(O_block_ptr, o_j.to(O_block_ptr.type.element_ty))
    tl.store(L_block_ptr, l_j.to(L_block_ptr.type.element_ty))


@torch.compile()
def flash_bwd_pytorch(Q, K, V, L, O, dO, is_causal):
    b, Nq, d = Q.shape
    scale = 1 / math.sqrt(d)

    D = torch.sum(dO * O, dim=-1)

    S = einsum(Q, K, "b Bq d, b Bk d -> b Bq Bk") * scale
    if is_causal:
        mask = torch.triu(torch.ones(Nq, Nq, device=Q.device, dtype=torch.bool), diagonal=1)
        S = S.masked_fill(mask, float("-inf"))

    P = torch.exp(S - L.unsqueeze(-1))  # Don't need running softmax as we have stored L

    dV = P.transpose(-2, -1) @ dO
    dP = dO @ V.transpose(-2, -1)
    dS = P * (dP - D.unsqueeze(-1)) * scale
    dQ = dS @ K  # Must be atomic add in triton kernel for correctness.
    dK = dS.transpose(-2, -1) @ Q

    return dQ, dK, dV

"""@triton.autotune(
    configs = [
        triton.Config(
            {"Q_TILE_SIZE": Q_TILE_SIZE, "K_TILE_SIZE":K_TILE_SIZE},
            num_warps=num_warps,
        )
        for Q_TILE_SIZE in [16, 32]
        for K_TILE_SIZE in [64, 128]
        for num_warps in [2, 4, 8]
    ],
    key = ["N_QUERIES", "N_KEYS", "d"]
)"""
@triton.jit
def flash_bwd_kv(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    D_ptr,
    dK_ptr,
    dV_ptr,
    dO_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_dkb,
    stride_dkk,
    stride_dkd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_dvb,
    stride_dvk,
    stride_dvd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_dob,
    stride_doq,
    stride_dod,
    stride_lb,
    stride_lq,
    stride_db,
    stride_dq,
    N_QUERIES,
    N_KEYS,
    scale,
    is_causal: tl.constexpr,
    d: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    # Program indices
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, d),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, d),
        order=(1, 0),
    )

    # Set up K_block_ptr, V_block_ptr, O_block_ptr, L_block_ptr similarly
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, d),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, d),
        order=(1, 0),
    )

    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_dkb,
        shape=(N_KEYS, d),
        strides=(stride_dkk, stride_dkd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, d),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, d),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, d),
        order=(1, 0),
    )

    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_dvb,
        shape=(N_KEYS, d),
        strides=(stride_dvk, stride_dvd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, d),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, d),
        strides=(stride_oq, stride_od),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, d),
        order=(1, 0),
    )

    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, d),
        strides=(stride_doq, stride_dod),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, d),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # Precomputing D
    # D = torch.sum(dO * O, dim=-1)
    
    Tk = N_KEYS // K_TILE_SIZE
    Tq = N_QUERIES // Q_TILE_SIZE

    if is_causal:
        initial_tile = key_tile_index * K_TILE_SIZE // Q_TILE_SIZE
        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE*initial_tile, 0))
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE*initial_tile,))
        dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE*initial_tile, 0))
        D_block_ptr = D_block_ptr.advance((Q_TILE_SIZE*initial_tile,))
        diag_tiles = max(K_TILE_SIZE // Q_TILE_SIZE, 1)

    else: 
        initial_tile = 0
        diag_tiles = 0
    
    end_diag = initial_tile + diag_tiles
    # implementing flash attention algo
    #for j in range(Tk):  # Tk
    K_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
    V_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
    dK_j = tl.zeros((K_TILE_SIZE, d), dtype=tl.float32)
    dV_j = tl.zeros((K_TILE_SIZE, d), dtype=tl.float32)
    
    
    col_idx = tl.arange(0, K_TILE_SIZE) + key_tile_index * K_TILE_SIZE
    Q_base = tl.arange(0, Q_TILE_SIZE)
    
    # Diagonals in separate loop:
    for i_diag in range(initial_tile, end_diag):  # 
        Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        L_i = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
        #O_i = tl.load(O_block_ptr, boundary_check=(0, 1), padding_option="zero")
        dO_i = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")
        D_i = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")
        S_ij = tl.dot(Q_i, tl.trans(K_j)) * scale  # compute pre-softmax attention

        # Using absolute position instead of relative
        row_idx = Q_base + i_diag*Q_TILE_SIZE
        causal_mask = row_idx[:, None] >= col_idx[None, :]
        S_ij = tl.where(causal_mask, S_ij, float("-inf"))
       
        P_ij = tl.exp(S_ij - L_i[:, None])  # Don't need running softmax as we have stored L

        dV_j += tl.dot(tl.trans(P_ij.to(dO_i.dtype)), dO_i)
        dP_ij = tl.dot(dO_i, tl.trans(V_j))
        dS_ij = P_ij * (dP_ij - D_i[:, None]) * scale
        

        
        dK_j += tl.dot(tl.trans(dS_ij).to(Q_i.dtype), Q_i)

        # advance Q, L, dO and D pointers
        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE,))
        dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE, 0))
        D_block_ptr = D_block_ptr.advance((Q_TILE_SIZE,))

    # Loop over <<< remaing tiles:
    for i in range(end_diag, Tq):  # 
        Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        L_i = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
        #O_i = tl.load(O_block_ptr, boundary_check=(0, 1), padding_option="zero")
        dO_i = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")
        D_i = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")
        S_ij = tl.dot(Q_i, tl.trans(K_j)) * scale  # compute pre-softmax attention

        P_ij = tl.exp(S_ij - L_i[:, None])  # Don't need running softmax as we have stored L

        dV_j += tl.dot(tl.trans(P_ij.to(dO_i.dtype)), dO_i)
        dP_ij = tl.dot(dO_i, tl.trans(V_j))
        dS_ij = P_ij * (dP_ij - D_i[:, None]) * scale
        

        
        dK_j += tl.dot(tl.trans(dS_ij).to(Q_i.dtype), Q_i)

        # advance Q, L, dO and D pointers
        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE,))
        dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE, 0))
        D_block_ptr = D_block_ptr.advance((Q_TILE_SIZE,))


    tl.store(dK_block_ptr, dK_j.to(dK_block_ptr.type.element_ty))
    tl.store(dV_block_ptr, dV_j.to(dV_block_ptr.type.element_ty))

"""@triton.autotune(
    configs = [
        triton.Config(
            {"Q_TILE_SIZE": Q_TILE_SIZE, "K_TILE_SIZE":K_TILE_SIZE},
            num_warps=num_warps,
        )
        for Q_TILE_SIZE in [64, 128, 256]
        for K_TILE_SIZE in [16, 32, 64]
        for num_warps in [2, 4, 8]
    ],
    key = ["N_QUERIES", "N_KEYS", "d"]
)"""
@triton.jit
def flash_bwd_dq(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    D_ptr,
    dQ_ptr,
    dO_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_dqb,
    stride_dqq,
    stride_dqd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_dob,
    stride_doq,
    stride_dod,
    stride_lb,
    stride_lq,
    stride_db,
    stride_dq,
    N_QUERIES,
    N_KEYS,
    scale,
    is_causal: tl.constexpr,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):

    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_index * stride_dqb,
        shape=(N_QUERIES, D),
        strides=(stride_dqq, stride_dqd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    # Set up K_block_ptr, V_block_ptr, O_block_ptr, L_block_ptr similarly
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )


    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )


    Q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    L = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
    #O_i = tl.load(O_block_ptr, boundary_check=(0, 1), padding_option="zero")
    dO = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")
    D_i = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")

    last_query_pos = (query_tile_index + 1) * Q_TILE_SIZE - 1
    num_k_tiles = last_query_pos // K_TILE_SIZE + 1 if is_causal else N_KEYS // K_TILE_SIZE


    if is_causal:
        start_diag_tile = query_tile_index * Q_TILE_SIZE // K_TILE_SIZE

    else: 
        start_diag_tile = num_k_tiles

    
    row_idx = tl.arange(0, Q_TILE_SIZE) + query_tile_index * Q_TILE_SIZE
    K_base = tl.arange(0, K_TILE_SIZE)
    # Diagonals in separate loop:
    dQ_acc = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    for j in range(start_diag_tile):
        K_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        S_ij = tl.dot(Q, tl.trans(K_j)) * scale
        P_ij = tl.exp(S_ij - L[:, None])  # Don't need running softmax as we have stored L
        dP_ij = tl.dot(dO, tl.trans(V_j))
        dS_ij = P_ij * (dP_ij - D_i[:, None]) * scale

        dQ_acc += tl.dot(dS_ij.to(K_j.dtype), K_j)

        # advance k- pointer
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    
    # Diagonal elements if masking
    for j_diag in range(start_diag_tile, num_k_tiles):
        K_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        # tl.device_print("k_j", k_j)
        S_ij = tl.dot(Q, tl.trans(K_j)) * scale
        # Do masking for this loop.
        col_idx = K_base + j_diag * K_TILE_SIZE
        causal_mask = row_idx[:, None] >= col_idx[None, :]
        S_ij = tl.where(causal_mask, S_ij, float("-inf"))
        P_ij = tl.exp(S_ij - L[:, None])  # Don't need running softmax as we have stored L
        dP_ij = tl.dot(dO, tl.trans(V_j))
        dS_ij = P_ij * (dP_ij - D_i[:, None]) * scale

        dQ_acc += tl.dot(dS_ij.to(K_j.dtype), K_j)

        # advance k- pointer
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))


    tl.store(dQ_block_ptr, dQ_acc.to(dQ_block_ptr.type.element_ty))



class TritonFlashAttentionAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q_ptr, K_ptr, V_ptr, is_causal: bool = False):
        b, N_QUERIES, D = Q_ptr.shape
        b, N_KEYS, D = K_ptr.shape
        scale = 1 / math.sqrt(D)
        ctx.is_causal = is_causal
        O_ptr = torch.empty((Q_ptr.shape), device=Q_ptr.device, dtype=Q_ptr.dtype)
        L_ptr = torch.empty((b, N_QUERIES), device=Q_ptr.device, dtype=Q_ptr.dtype)

        stride_qb, stride_qq, stride_qd = (Q_ptr.stride(0), Q_ptr.stride(1), Q_ptr.stride(2))
        stride_kb, stride_kk, stride_kd = (K_ptr.stride(0), K_ptr.stride(1), K_ptr.stride(2))
        stride_vb, stride_vk, stride_vd = (V_ptr.stride(0), V_ptr.stride(1), V_ptr.stride(2))
        stride_ob, stride_oq, stride_od = (O_ptr.stride(0), O_ptr.stride(1), O_ptr.stride(2))
        stride_lb, stride_lq = (L_ptr.stride(0), L_ptr.stride(1))

        Q_TILE_SIZE = 64
        K_TILE_SIZE = 64
        Tq = N_QUERIES // Q_TILE_SIZE

        grid = (Tq, b)  # launch independent batches and Q-tiles across SM's
        #grid = lambda META: (N_QUERIES // META['Q_TILE_SIZE'], b)
        flash_fwd_kernel[grid](
            Q_ptr,
            K_ptr,
            V_ptr,
            O_ptr,
            L_ptr,
            stride_qb,
            stride_qq,
            stride_qd,
            stride_kb,
            stride_kk,
            stride_kd,
            stride_vb,
            stride_vk,
            stride_vd,
            stride_ob,
            stride_oq,
            stride_od,
            stride_lb,
            stride_lq,
            N_QUERIES,
            N_KEYS,
            scale,
            is_causal,
            D,
            Q_TILE_SIZE, # comment out if autotuning.
            K_TILE_SIZE, # comment out if autotuning.
            
        )
        # In forward, before returning:
        #print(f"O_ptr dtype: {O_ptr.dtype}, Q_ptr dtype: {Q_ptr.dtype}")
        ctx.save_for_backward(Q_ptr, K_ptr, V_ptr, L_ptr, O_ptr)
        return O_ptr

    @staticmethod
    def backward(ctx, dO_ptr):
        Q_ptr, K_ptr, V_ptr, L_ptr, O_ptr = ctx.saved_tensors
        b, N_QUERIES, d = Q_ptr.shape
        b, N_KEYS, d = K_ptr.shape
        is_causal = ctx.is_causal
        scale = 1 / math.sqrt(d)
        # dQ, dK, dV = flash_bwd_pytorch(Q, K, V, L, O, dO, is_causal)

        dQ_ptr = torch.zeros((Q_ptr.shape), device=Q_ptr.device, dtype=Q_ptr.dtype)
        dK_ptr = torch.empty((K_ptr.shape), device=K_ptr.device, dtype=K_ptr.dtype)
        dV_ptr = torch.empty((V_ptr.shape), device=V_ptr.device, dtype=V_ptr.dtype)

        # Precomputing D
        #D_ptr = torch.empty((L_ptr.shape), device=L_ptr.device, dtype=L_ptr.dtype)
        D_ptr = torch.sum(dO_ptr * O_ptr, dim=-1)

        stride_qb, stride_qq, stride_qd = (Q_ptr.stride(0), Q_ptr.stride(1), Q_ptr.stride(2))
        stride_dqb, stride_dqq, stride_dqd = (dQ_ptr.stride(0), dQ_ptr.stride(1), dQ_ptr.stride(2))
        stride_kb, stride_kk, stride_kd = (K_ptr.stride(0), K_ptr.stride(1), K_ptr.stride(2))
        stride_dkb, stride_dkk, stride_dkd = (dK_ptr.stride(0), dK_ptr.stride(1), dK_ptr.stride(2))
        stride_vb, stride_vk, stride_vd = (V_ptr.stride(0), V_ptr.stride(1), V_ptr.stride(2))
        stride_dvb, stride_dvk, stride_dvd = (dV_ptr.stride(0), dV_ptr.stride(1), dV_ptr.stride(2))
        stride_ob, stride_oq, stride_od = (O_ptr.stride(0), O_ptr.stride(1), O_ptr.stride(2))
        stride_dob, stride_doq, stride_dod = (dO_ptr.stride(0), dO_ptr.stride(1), dO_ptr.stride(2))
        stride_lb, stride_lq = (L_ptr.stride(0), L_ptr.stride(1))
        stride_db, stride_dq = (D_ptr.stride(0), D_ptr.stride(1))


        # dQ-kernel --------
        Q_TILE_SIZE_dq = 64
        K_TILE_SIZE_dq = 64
        Tq = N_QUERIES // Q_TILE_SIZE_dq
        #grid = lambda META: (N_QUERIES // META['Q_TILE_SIZE'], b)  
        grid = (Tq, b) # launch independent batches and Q-tiles across SM's
        flash_bwd_dq[grid](
            Q_ptr,
            K_ptr,
            V_ptr,
            O_ptr,
            L_ptr,
            D_ptr,
            dQ_ptr,
            dO_ptr,
            stride_qb,
            stride_qq,
            stride_qd,
            stride_dqb,
            stride_dqq,
            stride_dqd,
            stride_kb,
            stride_kk,
            stride_kd,
            stride_vb,
            stride_vk,
            stride_vd,
            stride_ob,
            stride_oq,
            stride_od,
            stride_dob,
            stride_doq,
            stride_dod,
            stride_lb,
            stride_lq,
            stride_db,
            stride_dq,
            N_QUERIES,
            N_KEYS,
            scale,
            is_causal,
            d,
            Q_TILE_SIZE_dq, # comment out if autotuning.
            K_TILE_SIZE_dq, # comment out if autotuning.

        )


        # KV-kernel ---------------------------

        Q_TILE_SIZE = 64
        K_TILE_SIZE = 64
        Tk = N_KEYS // K_TILE_SIZE

        #grid = lambda META: (N_KEYS // META['K_TILE_SIZE'], b)  
        grid = (Tk, b) # launch independent batches and Q-tiles across SM's
        flash_bwd_kv[grid](
            Q_ptr,
            K_ptr,
            V_ptr,
            O_ptr,
            L_ptr,
            D_ptr,
            dK_ptr,
            dV_ptr,
            dO_ptr,
            stride_qb,
            stride_qq,
            stride_qd,
            stride_kb,
            stride_kk,
            stride_kd,
            stride_dkb,
            stride_dkk,
            stride_dkd,
            stride_vb,
            stride_vk,
            stride_vd,
            stride_dvb,
            stride_dvk,
            stride_dvd,
            stride_ob,
            stride_oq,
            stride_od,
            stride_dob,
            stride_doq,
            stride_dod,
            stride_lb,
            stride_lq,
            stride_db,
            stride_dq,
            N_QUERIES,
            N_KEYS,
            scale,
            is_causal,
            d,
            Q_TILE_SIZE, # comment out if autotuning.
            K_TILE_SIZE, # comment out if autotuning.
            num_stages=1,
            num_warps=4

        )


        # In backward, before returning:
        #print(f"dQ dtype: {dQ_ptr.dtype}, dK dtype: {dK_ptr.dtype}, dV dtype: {dV_ptr.dtype}")
        #print(f"Expected dtype: {Q_ptr.dtype}")
        return dQ_ptr, dK_ptr, dV_ptr, None


@torch.compile()
# @dynamo.disable
def scaled_dot_product_attention(Q, K, V, mask):
    d_k = Q.shape[-1]
    seq_len = Q.shape[-2]
    # print(f"{Q.shape=}  {K.shape=} | {V.shape=}")
    # Q^T K / sqrt(d_k)
    attn = einsum(Q, K, "b ... sq d_k, b ... sk d_k -> b ... sq sk") / math.sqrt(d_k)
    # apply mask if included
    if mask is not None:
        m = mask.to(bool)
        attn = attn.masked_fill(~mask, float("-inf"))
    result = einsum(softmax(x=attn, dimension=-1), V, "b ... sq sk, b ... sk d_v -> b ... sq d_v")

    return result


def test_timing_flash_forward_backward():
    n_heads = 16*128
    d_head = 64
    sequence_length = 512
    q, k, v = torch.randn(3, n_heads, sequence_length, d_head, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    flash = torch.compile(TritonFlashAttentionAutogradFunction.apply)

    def flash_forward_backward():
        o = flash(q, k, v, True)
        loss = o.sum()
        loss.backward()

    results = triton.testing.do_bench(flash_forward_backward, rep=1000, warmup=100)  # rep=10000, warmup=1000)
    print(results)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["seq_dim"],
        x_vals=[2**i for i in range(7, 10)],  # 16
        line_arg="attention_function",
        line_vals=["Triton_kernel", "FA_torch", "Torch.compile"],
        line_names=["Triton kernel", "nn.scaled_dot..", "torch.compile()"],
        styles=[("blue", "-"), ("green", "-"), ("red", "-")],
        ylabel="GB/s",
        plot_name="Attention",  # name for the plot. Used also as a file name for saving the plot.
        args={
            "batch_dim": 128,
            "num_heads": 16,
            "head_dim": 64,
            "dtype": torch.bfloat16,
            "is_causal": True,
            "direction": "both",
        },  # values for function arguments not in `x_names` and `y_name`
    )
)
def benchmark_attention(batch_dim, num_heads, seq_dim, head_dim, dtype, attention_function, is_causal, direction):
    """
    Based on the benchmarking sample from triton-tutorials:
    https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html
    """
    from cs336_basics.model import scaled_dot_product_attention as compiled_sdpa
    from torch.nn.functional import scaled_dot_product_attention as nn_sdpa

    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    Q = torch.randn(batch_dim, num_heads, seq_dim, head_dim, device=DEVICE, dtype=dtype, requires_grad=True)
    K = torch.randn(batch_dim, num_heads, seq_dim, head_dim, device=DEVICE, dtype=dtype, requires_grad=True)
    V = torch.randn(batch_dim, num_heads, seq_dim, head_dim, device=DEVICE, dtype=dtype, requires_grad=True)

    # mask = torch.tril(torch.ones((seq_dim, seq_dim), device=DEVICE, dtype=dtype))

    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)

    if attention_function == "Triton_kernel":
        #Flatten Q, K and V
        b, h, s, d = Q.shape
        Q_flat = Q.reshape(b * h, s, d)
        K_flat = K.reshape(b * h, s, d)
        V_flat = V.reshape(b * h, s, d)
        ms = test_wrapper(lambda: TritonFlashAttentionAutogradFunction.apply(Q_flat, K_flat, V_flat, is_causal), direction, Q, K, V)
    if attention_function == "FA_torch":
        ms = test_wrapper(lambda: nn_sdpa(Q, K, V, is_causal=is_causal), direction, Q, K, V)
    if attention_function == "Torch.compile":
        ms = test_wrapper(lambda: compiled_sdpa(Q, K, V, mask=None), direction, Q, K, V)

    # GBperSec = lambda ms: 2 * (Q.numel() + K.numel() + V.numel()) * Q.element_size() * 1e-9 / (ms * 1e-3)
    causal_scale = 1 / (1 + is_causal)
    if direction == "both":
        direction_scale = 3
    elif direction == "backward":
        direction_scale = 2
    else:
        direction_scale = 1
    tflops = (
        dtype.itemsize * batch_dim * num_heads * seq_dim * seq_dim * head_dim * causal_scale * direction_scale * 1e-12 / (ms * 1e-3)
    )

    return tflops


def test_wrapper(func, direction, Q, K, V):
    if direction == "forward":
        ms_fwd = triton.testing.do_bench(func)
        return ms_fwd
    elif direction == "backward":
        dO = torch.randn(Q.shape, device=Q.device, dtype=Q.dtype)
        ms_fwd = triton.testing.do_bench(func)
        ms_fwd_bwd = triton.testing.do_bench(lambda: func().backward(dO))
        return ms_fwd_bwd - ms_fwd  # Estimate backward time
    elif direction == "both":
        # Forward + backward
        dO = torch.randn(Q.shape, device=Q.device, dtype=Q.dtype)

        def fwd_bwd():
            Q.grad, K.grad, V.grad = None, None, None
            out = func()
            b, h, s, d = Q.shape
            out = out.reshape(b, h, s, d)
            out.backward(dO)

        ms_both = triton.testing.do_bench(fwd_bwd)
        return ms_both
    else:
        raise NotImplementedError('Function can only be run with direction = "forward", "backward" or "both".')


def test_timing_flash_forward_backward():
    n_heads = 16*128
    d_head = 64
    sequence_length = 512
    q, k, v = torch.randn(3, n_heads, sequence_length, d_head, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    flash = torch.compile(TritonFlashAttentionAutogradFunction.apply)

    def flash_forward_backward():
        o = flash(q, k, v, True)
        loss = o.sum()
        loss.backward()

    results = triton.testing.do_bench(flash_forward_backward, rep=1000, warmup=100)  # rep=10000, warmup=1000)
    print(results)


if __name__ == "__main__":
    benchmark_attention.run(show_plots=True, print_data=True)

    test_timing_flash_forward_backward() # 
    print("should give 3.40 roughly")
