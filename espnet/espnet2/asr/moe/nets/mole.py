import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRAExpert(nn.Module):
    """A single Low-Rank Adaptation (LoRA) expert."""
    def __init__(self, d_in, d_out, rank=8, alpha=1, lora_dropout_rate=0.0, **kwargs):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_dropout_rate = lora_dropout_rate

        if lora_dropout_rate > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout_rate)

        self.lora_A = nn.Parameter(torch.zeros(rank, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化LoRA参数"""
        # A用高斯分布初始化，B用零初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        """
        Forward pass for the LoRA expert.
        Note that this only returns the LoRA-adapted part of the output.
        """
        B,T,D = x.shape

        x = x.reshape(B*T, D)
        if self.lora_dropout_rate > 0.:
            x = self.lora_dropout(x)
        out = self.scaling * (x @ self.lora_A.transpose(0,1) @ self.lora_B.transpose(0, 1))
        out = out.reshape(B, T, -1)

        return out


class MoLE(nn.Module):
    def __init__(self, 
                 num_experts, d_in, d_out, 
                 rank=8, alpha=8.0, lora_dropout=0.0,
                 router_dropout=0.0,
                 global_router=False, 
                 use_dynamic_router=True,
                 **kwargs):
        super().__init__()
        self.num_experts = num_experts
        self.d_in = d_in
        self.d_out = d_out
        self.rank = rank
        self.alpha = alpha
        self.global_router = global_router
        self.fc = nn.Linear(d_in, d_out)

        if rank > 0:
            self.experts = nn.ModuleList([
            LoRAExpert(d_in, d_out, rank, alpha, lora_dropout) for _ in range(num_experts)
            ])
        else:
            self.experts = None
        
        self.use_dynamic_router = use_dynamic_router
        if self.global_router and self.use_dynamic_router:
            self.dynamic_router = nn.Linear(d_in, 2)
        self.router = nn.Linear(d_in, num_experts)
        self.router_dropout = nn.Dropout(router_dropout) if router_dropout > 0 else None

    def forward(self, x, global_weights=None):
        
        B, T, D = x.shape
        base_output = self.fc(x)  # (B, T, d_out)

        if self.rank <= 0  or self.experts is None:
            return base_output, torch.zeros(B, T, self.num_experts, device=x.device, dtype=x.dtype)
        
        router_logits = self.router(x)
        if self.router_dropout is not None:
            router_logits = self.router_dropout(router_logits)

        local_weights = F.softmax(router_logits, dim=-1)  # (B, T, num_experts)
        routing_weights = local_weights
        
        if global_weights is not None:
            if self.global_router and self.use_dynamic_router:
                dy = F.softmax(self.dynamic_router(x), dim=-1)
                routing_weights = dy[:, :, 0:1] * routing_weights + dy[:, :, 1:2] * global_weights
            elif self.global_router:
                routing_weights = routing_weights + global_weights
            else:
                routing_weights = routing_weights

        lora_output = torch.zeros_like(base_output)
        for i, expert in enumerate(self.experts):
            expert_weight = routing_weights[:, :, i:i+1]  # (B, T, 1)
            expert_out = expert(x)  # (B, T, d_out)
            lora_output += expert_weight * expert_out
        
        return base_output + lora_output, routing_weights

