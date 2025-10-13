import torch
import torch.nn as nn
import torch.nn.functional as F


class Module(nn.Module):
    def __init__(
        self,
        tau: float=4.0, 
        beta: float=0.25,
    ):
        super().__init__()

        self.tau = tau
        self.beta = beta

    def forward(self, scores):
        # numerator
        numerator = F.relu(scores) ** self.tau
        
        # denominator
        numerator_sum = numerator.sum(dim=-1, keepdim=True) + 1e-8
        denominator = numerator_sum ** self.beta
        
        # attention weight
        weights = numerator / denominator
        
        # stabilize -inf, +inf
        valid = torch.isfinite(weights)
        weights = weights.masked_fill(~valid, 0.0)
        
        return numerator / denominator