from typing import Optional
import torch
import torch.nn as nn
from .sampler import LognormalSampler, WeibullSampler
from . import simplex_proj_fn
from ..utils.constants import (
    SAMPLER_TYPE,
    SCORE_FN_TYPE,
)


class BayesianAttentionModules(nn.Module):
    def __init__(
        self,
        dim: int,
        hyper_approx: float=0.1,
        hyper_prior: float=1.0,
        tau: float=4.0, 
        beta: float=0.25,
        dropout: float=0.2,
        sampler_type: SAMPLER_TYPE="lognormal",
        score_fn_type: SCORE_FN_TYPE="concat",
    ):
        super().__init__()

        self.dim = dim
        self.hyper_approx = hyper_approx
        self.hyper_prior = hyper_prior
        self.tau = tau
        self.beta = beta
        self.dropout = dropout
        self.sampler_type = sampler_type
        self.score_fn_type = score_fn_type

        self._set_up_components()

    def forward(
        self, 
        Q: torch.Tensor,                    # (B,D)
        K: torch.Tensor,                    # (B,H,D)
        V: torch.Tensor,                    # (B,H,D)
        mask: Optional[torch.Tensor]=None,  # (B,H)
        sampling: bool=True,
    ):
        # Q: (B,D) -> (B,1,D) -> (B,H,D)
        Q_exp = Q.unsqueeze(1).expand_as(K)

        # Sampling attn score: (B,H)
        samples, psi, kl = self.sampler(Q_exp, K, mask)

        # Masking: (B,H) or (H,) -> (B,H)
        if sampling==True:
            scores = samples.masked_fill(~mask, float('-inf'))
        else:
            scores = psi.masked_fill(~mask, float('-inf'))

        # Simplex projection: (B,H) -> (B,H)
        weights = self.simplex_proj_fn(scores)

        # Context vector: (B,H) x (B,H,D) -> (B,D)
        context = torch.sum(weights.unsqueeze(-1) * V, dim=1)

        return context, kl

    def _set_up_components(self):
        self._create_sampler()
        self._create_simplex_proj_fn()
    
    def _create_sampler(self):
        kwargs = dict(
            dim=self.dim, 
            score_fn_type=self.score_fn_type, 
            hyper_approx=self.hyper_approx, 
            hyper_prior=self.hyper_prior, 
            dropout=self.dropout,
        )

        if self.sampler_type=='lognormal':
            self.sampler = LognormalSampler(**kwargs)
        elif self.sampler_type=='weibull':
            self.sampler = WeibullSampler(**kwargs)
        else:
            raise ValueError(f"Invalid sampler_type: {self.sampler_type}")

    def _create_simplex_proj_fn(self):
        kwargs = dict(
            tau=self.tau, 
            beta=self.beta, 
        )
        self.simplex_proj_fn = simplex_proj_fn.Module(**kwargs)