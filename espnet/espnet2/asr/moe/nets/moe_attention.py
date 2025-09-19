#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Multi-Head Attention layer definition."""

import logging
import math

import torch
from torch import nn

from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet2.asr.moe.nets.mole import MoLE

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input
except Exception as e:
    print(f"Failed to import Flash Attention, using ESPnet default: {e}")


class MOE_MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
        qk_norm (bool): Normalize q and k before dot product.
        use_flash_attn (bool): Use flash_attn implementation.
        causal (bool): Apply causal attention.
        cross_attn (bool): Cross attention instead of self attention.
        use_sdpa (bool): Use PyTorch's scaled dot product attention.

    """

    def __init__(
        self,
        n_head,
        n_feat,
        dropout_rate,
        qk_norm=False,
        use_flash_attn=False,
        causal=False,
        cross_attn=False,
        use_sdpa=False,
        num_experts=3, lora_rank=8, lora_alpha=1.0, lora_dropout=0.0,
        router_dropout=0.0, global_router=True, use_dynamic_router=True
    ):
        """Construct an MultiHeadedAttention object."""
        super(MOE_MultiHeadedAttention, self).__init__()

        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = MoLE(num_experts, n_feat, n_feat, 
                        lora_rank, lora_alpha, lora_dropout,
                        router_dropout, global_router, use_dynamic_router)
        self.linear_k = MoLE(num_experts, n_feat, n_feat, 
                        lora_rank, lora_alpha, lora_dropout,
                        router_dropout, global_router, use_dynamic_router)
        self.linear_v = MoLE(num_experts, n_feat, n_feat, 
                        lora_rank, lora_alpha, lora_dropout,
                        router_dropout, global_router, use_dynamic_router)
        self.linear_out = MoLE(num_experts, n_feat, n_feat, 
                        lora_rank, lora_alpha, lora_dropout,
                        router_dropout, global_router, use_dynamic_router)
        self.attn = None
        self.dropout = (
            nn.Dropout(p=dropout_rate) if not use_flash_attn else nn.Identity()
        )
        self.dropout_rate = dropout_rate

        # LayerNorm for q and k
        self.q_norm = LayerNorm(self.d_k) if qk_norm else nn.Identity()
        self.k_norm = LayerNorm(self.d_k) if qk_norm else nn.Identity()

        self.use_flash_attn = use_flash_attn
        self.causal = causal  # only used with flash_attn
        self.cross_attn = cross_attn  # only used with flash_attn

        self.use_sdpa = use_sdpa

    def forward_qkv(self, query, key, value, expand_kv=False, global_router=None):
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            expand_kv (bool): Used only for partially autoregressive (PAR) decoding.

        Returns:
            torch.Tensor: Transformed query tensor (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor (#batch, n_head, time2, d_k).

        """
        n_batch = query.size(0)
        q, q_weights = self.linear_q(query, global_router)
        q = q.view(n_batch, -1, self.h, self.d_k)

        if expand_kv:
            k_shape = key.shape
            k1, k_weights = self.linear_k(key[:1, :, :], global_router)
            k = (
                k1
                .expand(n_batch, k_shape[1], k_shape[2])
                .view(n_batch, -1, self.h, self.d_k)
            )
            v_shape = value.shape
            v1, v_weights = self.linear_v(value[:1, :, :], global_router)
            v = (
                v1
                .expand(n_batch, v_shape[1], v_shape[2])
                .view(n_batch, -1, self.h, self.d_k)
            )
        else:
            k, k_weights = self.linear_k(key, global_router)
            k = k.view(n_batch, -1, self.h, self.d_k)
            v, v_weights = self.linear_v(value, global_router)
            v = v.view(n_batch, -1, self.h, self.d_k)

        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        q = self.q_norm(q)
        k = self.k_norm(k)

        return q, k, v

    def forward_attention(self, value, scores, mask, global_router=None):
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2).

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            min_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)

        res, res_states = self.linear_out(x, global_router)
        return  res # (batch, time1, d_model)

    def forward(self, query, key, value, mask, expand_kv=False, global_router=None):
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
            expand_kv (bool): Used only for partially autoregressive (PAR) decoding.
                When set to `True`, `Linear` layers are computed only for the first
                batch. This is useful to reduce the memory usage during decoding
                when the batch size is #beam_size x #mask_count, which can be large.
                Typically, in single waveform inference of PAR, `Linear` layers
                should not be computed for all batches for source-attention.

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
        """
        # Use PyTorch's Scaled Dot Product Attention implementation
        if getattr(self, "use_sdpa", False):
            q, k, v = self.forward_qkv(query, key, value, expand_kv, global_router)

            # The shape of mask must be broadcastable to the shape of attention weights
            out = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                mask.unsqueeze(1) if mask is not None else None,
                dropout_p=self.dropout_rate if self.training else 0.0,
            )  # (batch, head, time1, d_k)

            out = out.transpose(1, 2)  # (batch, time1, head, d_k)
            out = out.reshape(out.shape[0], out.shape[1], -1)  # (batch, time1, d_model)
            res, res_states = self.linear_out(out, global_router)
            return res  # (batch, time1, d_model)

        # Use Flash Attention implementation
        if self.use_flash_attn:
            try:
                # In the causal case, the last row will be the key mask
                key_nonpad_mask = mask[:, -1, :]  # (#batch, time2)
                if self.cross_attn:
                    # For cross attention, we do not know the query padding
                    query_nonpad_mask = torch.ones(
                        size=query.shape[:2], dtype=torch.bool, device=query.device
                    )
                else:
                    query_nonpad_mask = key_nonpad_mask

                if key_nonpad_mask.eq(0).any():
                    # Use variable length implementation if padded
                    q, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(
                        query, query_nonpad_mask
                    )[:4]
                    k, indices_k, cu_seqlens_k, max_seqlen_k = unpad_input(
                        key, key_nonpad_mask
                    )[:4]
                    v, _, _, _ = unpad_input(value, key_nonpad_mask)[:4]
                    
                    q, q_weights = self.linear_q(q, global_router)
                    q = q.reshape(-1, self.h, self.d_k)
                    k, k_weights = self.linear_k(k, global_router)
                    k = k.reshape(-1, self.h, self.d_k)
                    v, v_weights = self.linear_v(v, global_router)
                    v = v.reshape(-1, self.h, self.d_k)

                    q = self.q_norm(q)
                    k = self.k_norm(k)

                    out = flash_attn_varlen_func(
                        q,
                        k,
                        v,
                        cu_seqlens_q,
                        cu_seqlens_k,
                        max_seqlen_q,
                        max_seqlen_k,
                        dropout_p=self.dropout_rate if self.training else 0.0,
                        causal=self.causal,
                    )  # (total, nheads, headdim)

                    out = out.reshape(out.shape[0], -1)
                    out,out_weights = self.linear_out(out, global_router)

                    out = pad_input(out, indices_q, query.shape[0], query.shape[1])
                    return out

                else:
                    # Use fixed length implementation if not padded,
                    # which is faster than the variable length implementation
                    del key_nonpad_mask
                    q, k, v = self.forward_qkv(query, key, value, False, global_router)

                    out = flash_attn_func(
                        q.transpose(1, 2),
                        k.transpose(1, 2),
                        v.transpose(1, 2),
                        dropout_p=self.dropout_rate if self.training else 0.0,
                        causal=self.causal,
                    )  # (batch_size, seqlen, nheads, headdim)
                    del q, k, v

                    out = out.reshape(out.shape[0], out.shape[1], -1)
                    out, out_weights = self.linear_out(out, global_router)
                    return out

            except Exception as e:
                logging.warning(
                    f"Flash Attention failed, falling back to default attention: {e}"
                )
                self.use_flash_attn = False

        # Fall back to the default implementation
        q, k, v = self.forward_qkv(query, key, value, expand_kv, global_router)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask, global_router=global_router)


class MOE_LegacyRelPositionMultiHeadedAttention(MOE_MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding (old version).

    Details can be found in https://github.com/espnet/espnet/pull/2816.

    Paper: https://arxiv.org/abs/1901.02860

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
        zero_triu (bool): Whether to zero the upper triangular part of attention matrix.

    """

    def __init__(self, n_head, n_feat, dropout_rate, zero_triu=False,
                 num_experts=3, lora_rank=8, lora_alpha=1.0, lora_dropout=0.0,
                 router_dropout=0.0, global_router=True, use_dynamic_router=True):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate, 
                         False, False, False, False, False,
                         num_experts, lora_rank, lora_alpha, lora_dropout,
                         router_dropout, global_router)
        self.zero_triu = zero_triu
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x):
        """Compute relative positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, head, time1, time2).

        Returns:
            torch.Tensor: Output tensor.

        """
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)

        if self.zero_triu:
            ones = torch.ones((x.size(2), x.size(3)))
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(self, query, key, value, pos_emb, mask, global_router=None):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            pos_emb (torch.Tensor): Positional embedding tensor (#batch, time1, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q, k, v = self.forward_qkv(query, key, value, False, global_router)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, time1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, head, time1, time1)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(
            self.d_k
        )  # (batch, head, time1, time2)

        return self.forward_attention(v, scores, mask, global_router=global_router)


class MOE_RelPositionMultiHeadedAttention(MOE_MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding (new implementation).

    Details can be found in https://github.com/espnet/espnet/pull/2816.

    Paper: https://arxiv.org/abs/1901.02860

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
        zero_triu (bool): Whether to zero the upper triangular part of attention matrix.

    """

    def __init__(self, n_head, n_feat, dropout_rate, zero_triu=False,
                 num_experts=3, lora_rank=8, lora_alpha=1.0, lora_dropout=0.0,
                 router_dropout=0.0, global_router=True, use_dynamic_router=True):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate, 
                         False, False, False, False,False,
                         num_experts, lora_rank, lora_alpha, lora_dropout,
                         router_dropout, global_router)
        self.zero_triu = zero_triu
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x):
        """Compute relative positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, head, time1, 2*time1-1).
            time1 means the length of query vector.

        Returns:
            torch.Tensor: Output tensor.

        """
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)[
            :, :, :, : x.size(-1) // 2 + 1
        ]  # only keep the positions from 0 to time2

        if self.zero_triu:
            ones = torch.ones((x.size(2), x.size(3)), device=x.device)
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(self, query, key, value, pos_emb, mask, global_router=None):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            pos_emb (torch.Tensor): Positional embedding tensor
                (#batch, 2*time1-1, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q, k, v = self.forward_qkv(query, key, value, False, global_router)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, 2*time1-1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, head, time1, 2*time1-1)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(
            self.d_k
        )  # (batch, head, time1, time2)

        return self.forward_attention(v, scores, mask, global_router=global_router)
