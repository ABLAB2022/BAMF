import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from ..bam.model import BayesianAttentionModules
from ..utils.constants import (
    SAMPLER_TYPE,
    SCORE_FN_TYPE,
)


class Module(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_factors: int,
        user_hist: torch.Tensor,
        hyper_approx: float=0.1,
        hyper_prior: float=1.0,
        tau: float=4.0,
        beta: float=0.25,
        dropout: float=0.2,
        sampler_type: SAMPLER_TYPE="lognormal",
        score_fn_type: SCORE_FN_TYPE="concat",
    ):
        super().__init__()

        # attr dictionary for load
        self.init_args = locals().copy()
        del self.init_args["self"]
        del self.init_args["__class__"]

        # device setting
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(DEVICE)

        # global attr
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.hyper_approx = hyper_approx
        self.hyper_prior = hyper_prior
        self.tau = tau
        self.beta = beta
        self.dropout = dropout
        self.sampler_type = sampler_type
        self.score_fn_type = score_fn_type
        self.register_buffer(
            name="user_hist", 
            tensor=user_hist,
        )

        # generate layers
        self._set_up_components()

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        user_idx: (B,)
        item_idx: (B,)
        """
        return self.score(user_idx, item_idx, True)

    @torch.no_grad()
    def predict(
        self, 
        user_idx: torch.Tensor,
        item_idx: torch.Tensor, 
    ):
        logit, kl = self.score(user_idx, item_idx, False)
        prob = torch.sigmoid(logit).squeeze(-1)
        return prob

    def score(self, user_idx, item_idx, stochastic):
        pred_vector, kl = self.gmf(user_idx, item_idx, stochastic)
        logit = self.pred_layer(pred_vector).squeeze(-1)
        return logit, kl

    def gmf(self, user_idx, item_idx, stochastic):
        user_embed_slice, item_embed_slice, kl = self.rep(user_idx, item_idx, stochastic)
        pred_vector = user_embed_slice * item_embed_slice
        return pred_vector, kl

    def rep(self, user_idx, item_idx, stochastic):
        # id embedding
        user_embed_slice_id, target_embed_slice_id, hist_embed_slice_id, mask = self._embed_slicer(user_idx, item_idx)
        
        # history embedding
        kwargs = dict(
            refer=hist_embed_slice_id, 
            mask=mask, 
            stochastic=stochastic,
        )
        user_embed_slice_hist, kl_u = self.user_hist_embed_generator(
            **kwargs,
            query=user_embed_slice_id,
        )
        item_embed_slice_hist, kl_i = self.user_hist_embed_generator(
            **kwargs,
            query=target_embed_slice_id,
        )

        # element-wise product & layernorm
        user_embed_slice = self.norm_u(user_embed_slice_hist * user_embed_slice_id)
        item_embed_slice = self.norm_i(item_embed_slice_hist * target_embed_slice_id)

        return user_embed_slice, item_embed_slice, (kl_u + kl_i)/2

    def user_hist_embed_generator(self, query, refer, mask, stochastic):
        kwargs = dict(
            Q=query,
            K=refer,
            V=refer,
            mask=mask,
            stochastic=stochastic,
        )
        return self.bam_u(**kwargs)

    def item_hist_embed_generator(self, query, refer, mask, stochastic):
        kwargs = dict(
            Q=query,
            K=refer,
            V=refer,
            mask=mask,
            stochastic=stochastic,
        )
        return self.bam_i(**kwargs)

    def _embed_slicer(self, user_idx, item_idx):
        kwargs = dict(
            target_hist=self.user_hist, 
            target_idx=user_idx, 
            counterpart_padding_idx=self.n_items,
        )
        refer_idx = self._hist_idx_slicer(**kwargs)

        kwargs = dict(
            hist_idx_slice=refer_idx,
            counterpart_idx=item_idx, 
            counterpart_padding_idx=self.n_items,
        )
        mask = self._mask_generator(**kwargs)

        # get embeddings
        user_embed_slice        = self.user_embed(user_idx)
        target_embed_slice_raw  = self.item_embed(item_idx)
        hist_embed_slice_raw    = self.item_embed(refer_idx)        # (B, L, D)

        # linear transformation
        target_embed_slice      = self.W_i(target_embed_slice_raw)
        hist_embed_slice        = self.W_h(hist_embed_slice_raw)   # (B, L, D)

        return user_embed_slice, target_embed_slice, hist_embed_slice, mask

    def _mask_generator(self, hist_idx_slice, counterpart_idx, counterpart_padding_idx):
        # mask to current target item from history
        marking_counterpart_idx = hist_idx_slice == counterpart_idx.unsqueeze(1)
        # mask to padding
        marking_padding_idx = hist_idx_slice == counterpart_padding_idx
        # final mask
        mask = ~(marking_counterpart_idx | marking_padding_idx)
        return mask

    def _hist_idx_slicer(self, target_hist, target_idx, counterpart_padding_idx):
        # target hist slice
        hist_idx_slice = target_hist[target_idx]
        # calculate max hist in batch
        lengths = (hist_idx_slice != counterpart_padding_idx).sum(dim=1)
        max_len = lengths.max().item()
        # drop padding values
        hist_idx_slice_trunc = hist_idx_slice[:, :max_len]
        return hist_idx_slice_trunc

    def _set_up_components(self):
        self._create_modules()
        self._create_embeddings()
        self._init_embeddings()
        self._create_layers()

    def _create_modules(self):
        kwargs = dict(
            dim=self.n_factors, 
            sampler_type=self.sampler_type,
            score_fn_type=self.score_fn_type,
            hyper_approx=self.hyper_approx,
            hyper_prior=self.hyper_prior,
            tau=self.tau,
            beta=self.beta,
            dropout=self.dropout,
        )
        self.bam_u = BayesianAttentionModules(**kwargs)
        self.bam_i = BayesianAttentionModules(**kwargs)

    def _create_embeddings(self):
        self.user_embed = nn.Embedding(
            num_embeddings=self.n_users+1, 
            embedding_dim=self.n_factors,
            padding_idx=self.n_users,
        )
        self.item_embed = nn.Embedding(
            num_embeddings=self.n_items+1, 
            embedding_dim=self.n_factors,
            padding_idx=self.n_items,
        )

    def _init_embeddings(self):
        nn.init.normal_(self.user_embed.weight, std=0.01)
        nn.init.normal_(self.item_embed.weight, std=0.01)

    def _create_layers(self):
        self.W_i = nn.Linear(
            in_features=self.n_factors,
            out_features=self.n_factors,
            bias=False,
        )
        self.W_h = nn.Linear(
            in_features=self.n_factors,
            out_features=self.n_factors,
            bias=False,
        )

        self.norm_u = nn.LayerNorm(self.n_factors)
        self.norm_i = nn.LayerNorm(self.n_factors)

        self.pred_layer = nn.Linear(
            in_features=self.n_factors,
            out_features=1,
        )