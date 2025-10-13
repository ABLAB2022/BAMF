import torch
import torch.nn as nn


class Concat(nn.Module):
    def __init__(
        self,
        dim: int,
        dropout: float,
    ):
        super().__init__()

        self.dim = dim
        self.dropout = dropout

        self._set_up_components()

    def forward(
        self, 
        Q: torch.Tensor, 
        K: torch.Tensor,
    ):
        """
        Q: (B,K,D)
        K: (B,K,D)
        """
        # (B,K,2D)
        QK_cat = torch.cat([Q, K], dim=-1)
        # (B,K,2D) -> (B,K,1) -> (B,K)
        scores = self.score_fn(QK_cat).squeeze(-1)
        return scores

    def _set_up_components(self):
        self._create_layers()

    def _create_layers(self):
        self.score_fn = nn.Sequential(
            nn.Linear(self.dim * 2, self.dim),
            nn.LayerNorm(self.dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),

            nn.Linear(self.dim, 1),
        )


class Prod(nn.Module):
    def __init__(
        self,
        dim: int,
        dropout: float,
    ):
        super().__init__()

        self.dim = dim
        self.dropout = dropout

        self._set_up_components()

    def forward(
        self, 
        Q: torch.Tensor, 
        K: torch.Tensor,
    ):
        """
        Q: (B,K,D)
        K: (B,K,D)
        """
        # (B,K,D)
        QK_prod = Q * K
        # (B,K,D) -> (B,K,1) -> (B,K)
        scores = self.score_fn(QK_prod).squeeze(-1)
        return scores

    def _set_up_components(self):
        self._create_layers()

    def _create_layers(self):
        self.score_fn = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.LayerNorm(self.dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),

            nn.Linear(self.dim, 1),
        )