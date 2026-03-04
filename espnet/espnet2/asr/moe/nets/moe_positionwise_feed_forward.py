#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Positionwise feed forward layer definition."""

import torch
from espnet2.asr.moe.nets.mole import MoLE

class MOE_PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim, hidden_units, dropout_rate, activation=torch.nn.ReLU(),
                 num_experts=3, lora_rank=8, lora_alpha=1.0, lora_dropout=0.0,
                 router_dropout=0.0, global_router=True, use_dynamic_router=True,
                 topk=1, aux_loss_coef=0.01, d_global=256,):
        """Construct an PositionwiseFeedForward object."""
        super(MOE_PositionwiseFeedForward, self).__init__()
        self.w_1 = MoLE(num_experts, idim, hidden_units, 
                        lora_rank, lora_alpha, lora_dropout,
                        router_dropout, global_router, use_dynamic_router,
                        topk, aux_loss_coef, d_global)
        self.w_2 = MoLE(num_experts, hidden_units, idim,
                        lora_rank, lora_alpha, lora_dropout,
                        router_dropout, global_router, use_dynamic_router,
                        topk, aux_loss_coef, d_global)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.activation = activation

    def forward(self, x, global_router=None):
        """Forward function."""
        balance_loss = 0.0
        x1, weights_state1 = self.w_1(x, global_router)
        if weights_state1[1] is not None:
            balance_loss += weights_state1[1]
        x1 = self.dropout(self.activation(x1))
        x2, weights_state2 = self.w_2(x1, global_router)
        if weights_state2[1] is not None:
            balance_loss += weights_state2[1]

        return [x2, balance_loss]
