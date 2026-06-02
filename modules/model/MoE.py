import torch
from torch import nn as nn
from torch.nn import functional as F

class ConfidenceMoELayer(nn.Module):
    def __init__(self, dim, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        # Expert Network
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.LeakyReLU()
            ) for _ in range(num_experts)
        ])
        # Gate Network
        self.gate = nn.Linear(dim, num_experts)

    def forward(self, features, rel_embs):
        #batch_size, ent_num, dim = features.size()

        gates = F.softmax(self.gate(features), dim=-1)  # [batch_size, ent_num, num_experts]

        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(features)  # [batch_size, ent_num, dim]
            confidence = torch.multiply(expert_out, rel_embs).sum(dim=-1)  # [batch_size, ent_num]
            expert_outputs.append(confidence.unsqueeze(-1))

        expert_outputs = torch.cat(expert_outputs, dim=-1)  # [batch_size, ent_num, num_experts]
        edge_weight = torch.sum(gates * expert_outputs, dim=-1)  # [batch_size, ent_num]
        
        return edge_weight