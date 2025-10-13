import torch
import torch.nn as nn
from torch.distributions import LogNormal, Weibull, Gamma
from ..utils.constants import SCORE_FN_TYPE
from .score_fn import Concat, Prod
from .masked_kl import masked_kl_fn


class LognormalSampler(nn.Module):
    def __init__(
        self,
        dim,
        hyper_approx: float=0.1,
        hyper_prior: float=1.0,
        dropout: float=0.2,
        score_fn_type: SCORE_FN_TYPE="concat",
    ):
        super().__init__()
    
        self.dim = dim
        self.hyper_approx = hyper_approx
        self.hyper_prior = hyper_prior
        self.dropout = dropout
        self.score_fn_type = score_fn_type

        self._set_up_components()

    def forward(self, Q, K, mask):
        approx, psi = self.approx(Q, K)
        samples = approx.rsample()
        prior = self.prior(K)

        kwargs = dict(
            approx=approx,
            prior=prior,
            mask=mask.expand_as(samples),
        )
        kl = masked_kl_fn(**kwargs)

        return samples, psi, kl

    def approx(self, Q, K):
        psi = self.approx_exp_fn(Q, K)
        sigma = torch.full_like(psi, self.hyper_approx)
        mu = psi - 0.5 * (sigma ** 2)
        dist = LogNormal(mu, sigma)
        return dist, psi

    def prior(self, K):
        psi = self.prior_exp_fn(K).squeeze(-1)
        sigma = torch.full_like(psi, self.hyper_prior)
        mu = psi - 0.5 * (sigma ** 2)
        dist = LogNormal(mu, sigma)
        return dist

    def _set_up_components(self):
        self._create_approx_exp_fn()
        self._create_prior_exp_fn()

    def _create_approx_exp_fn(self):
        kwargs = dict(
            dim=self.dim, 
            dropout=self.dropout,
        )

        if self.score_fn_type=="concat":
            self.approx_exp_fn = Concat(**kwargs)
        elif self.score_fn_type=="prod":
            self.approx_exp_fn = Prod(**kwargs)
        else:
            raise ValueError(f"Invalid score_fn_type: {self.score_fn_type}")

    def _create_prior_exp_fn(self):
        self.prior_exp_fn = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.LayerNorm(self.dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.dim, 1),
            nn.Softmax(dim=-1),
        )


class WeibullSampler(nn.Module):
    def __init__(
        self,
        dim,
        hyper_approx: float=0.1,
        hyper_prior: float=1.0,
        dropout: float=0.2,
        score_fn_type: SCORE_FN_TYPE="concat",
    ):
        super().__init__()
    
        self.dim = dim
        self.score_fn_type = score_fn_type
        self.hyper_approx = hyper_approx
        self.hyper_prior = hyper_prior
        self.dropout = dropout
        
        self._set_up_components()

    def forward(self, Q, K, mask):
        approx, psi = self.approx(Q, K)
        samples = approx.rsample()
        prior = self.prior(K)

        dist = dict(
            approx=approx,
            prior=prior,
            mask=mask.expand_as(samples),
        )
        kl = masked_kl_fn(**dist)

        return samples, psi, kl

    def approx(self, Q, K):
        psi = self.approx_exp_fn(Q, K)
        k_ = torch.full_like(psi, self.hyper_approx)
        lambda_ = torch.exp(psi) / torch.exp(torch.lgamma(1 + 1.0 / k_))
        dist = Weibull(lambda_, k_)
        return dist, psi

    def prior(self, K):
        psi = self.prior_exp_fn(K).squeeze(-1)
        beta = torch.full_like(psi, self.hyper_prior)
        alpha = torch.exp(psi) * beta
        dist = Gamma(alpha, beta)
        return dist

    def _set_up_components(self):
        self._create_approx_exp_fn()
        self._create_prior_exp_fn()

    def _create_approx_exp_fn(self):
        kwargs = dict(
            dim=self.dim, 
            dropout=self.dropout,
        )

        if self.score_fn_type=="concat":
            self.approx_exp_fn = Concat(**kwargs)
        elif self.score_fn_type=="prod":
            self.approx_exp_fn = Prod(**kwargs)
        else:
            raise ValueError(f"Invalid score_fn_type: {self.score_fn_type}")

    def _create_prior_exp_fn(self):
        self.prior_exp_fn = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.LayerNorm(self.dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.dim, 1),
            nn.Softmax(dim=-1),
        )