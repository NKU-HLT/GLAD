import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRAExpert(nn.Module):
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
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
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
                 topk=1,
                 aux_loss_coef=0.01,
                 d_global=256,
                 **kwargs):
        super().__init__()
        self.num_experts = num_experts
        self.d_in = d_in
        self.d_out = d_out
        self.rank = rank
        self.alpha = alpha
        self.global_router = global_router
        self.fc = nn.Linear(d_in, d_out)
        self.topk = topk
        self.aux_loss_coef = aux_loss_coef

        if rank > 0:
            self.experts = nn.ModuleList([
            LoRAExpert(d_in, d_out, rank, alpha, lora_dropout) for _ in range(num_experts)
            ])
        else:
            self.experts = None
        
        self.use_dynamic_router = use_dynamic_router
        if self.global_router:
            self.global_router_linear = nn.Linear(d_global, self.num_experts, bias=False)
            
        if self.global_router and self.use_dynamic_router:
            self.dynamic_router = nn.Linear(d_in, 2)
        self.router = nn.Linear(d_in, num_experts, bias=False)
        # self.router_dropout = nn.Dropout(router_dropout) if router_dropout > 0 else None

    def forward(self, x, global_weights=[None, None]):
        
        B, T, D = x.shape
        base_output = self.fc(x)  # (B, T, d_out)

        mask = global_weights[0] # (B, T, 1)
        global_weights = global_weights[1]
        
        if self.rank <= 0 or self.experts is None:
            return base_output, [torch.zeros(B, T, self.num_experts, device=x.device, dtype=x.dtype), 0.0]
        
        router_logits = self.router(x)
        
        valid_mask = None
        if mask is not None:
            valid_mask = mask.transpose(1, 2)
            if valid_mask.dtype != torch.bool:
                valid_mask = (valid_mask > 0.5)
            router_logits = router_logits.masked_fill(~valid_mask, -1e9)

        local_probs = F.softmax(router_logits, dim=-1)  # (B, T, num_experts)
        
        current_aux_loss = 0.0
        
        current_aux_loss += self._compute_load_balancing_loss(
            local_probs, valid_mask
        )
        
        local_weights = self.topk_compute(local_probs, self.topk)
        final_weights = local_weights
        
        if global_weights is not None:
            g_logits = self.global_router_linear(global_weights)
            if valid_mask is not None:
                g_logits = g_logits.masked_fill(~valid_mask, -1e9)
            global_probs = F.softmax(g_logits, dim=-1)
            
            current_aux_loss += self._compute_load_balancing_loss(
                global_probs, valid_mask
            )
            
            global_weights_sparse = self.topk_compute(global_probs, self.topk)

            if self.global_router and self.use_dynamic_router:
                dy = F.softmax(self.dynamic_router(x), dim=-1)
                final_weights = dy[:, :, 0:1] * local_weights + dy[:, :, 1:2] * global_weights_sparse
            elif self.global_router:
                final_weights = local_weights + global_weights_sparse
            else:
                final_weights = local_weights

        lora_output = torch.zeros_like(base_output)
        
        for i, expert in enumerate(self.experts):
            expert_weight = final_weights[:, :, i:i+1]  # (B, T, 1)
            if expert_weight.sum() == 0:
                continue
            expert_out = expert(x)  # (B, T, d_out)
            lora_output += expert_weight * expert_out
        
        return base_output + lora_output, [final_weights, current_aux_loss]

    def _compute_load_balancing_loss(self, probs, valid_mask=None):
        probs_flat = probs.view(-1, self.num_experts)  # (B*T, num_experts)
        
        if valid_mask is not None:
            mask_flat = valid_mask.contiguous().view(-1)  # (B*T,)
            total_valid_tokens = mask_flat.sum().float() + 1e-10
            
            mean_probs = (probs_flat * mask_flat.unsqueeze(-1).float()).sum(dim=0) / total_valid_tokens
            
            _, topk_indices = torch.topk(probs_flat, self.topk, dim=-1)  # (B*T, k)
            
            expert_mask = torch.zeros_like(probs_flat)
            expert_mask.scatter_(1, topk_indices, 1.0)
            
            expert_counts = (expert_mask * mask_flat.unsqueeze(-1).float()).sum(dim=0)  # (num_experts,)

            fraction_per_expert = expert_counts / (total_valid_tokens * self.topk)
                
        else:
            total_tokens = probs_flat.shape[0]

            mean_probs = probs_flat.mean(dim=0)  # (num_experts,)

            _, topk_indices = torch.topk(probs_flat, self.topk, dim=-1)  # (B*T, k)
            expert_mask = torch.zeros_like(probs_flat)
            expert_mask.scatter_(1, topk_indices, 1.0)
            
            expert_counts = expert_mask.sum(dim=0)  # (num_experts,)
            fraction_per_expert = expert_counts / (total_tokens * self.topk)

        aux_loss = self.aux_loss_coef * self.num_experts * (fraction_per_expert * mean_probs).sum()
        
        return aux_loss
    
    def topk_compute(self, probs, k):
        if k >= self.num_experts:
            return probs

        topk_vals, topk_indices = torch.topk(probs, k, dim=-1)

        topk_vals_normalized = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-10)

        mask_weights = torch.zeros_like(probs)
        mask_weights.scatter_(-1, topk_indices, topk_vals_normalized)
        
        return mask_weights