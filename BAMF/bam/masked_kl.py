import torch
from torch.distributions import kl_divergence


def masked_kl_fn(
    approx: torch.distributions, 
    prior: torch.distributions, 
    mask: torch.Tensor,
):
    # compute kl divergence
    kl_tensor = kl_divergence(approx, prior)

    # masking
    mask = mask.to(kl_tensor.device)
    kl_tensor_masked = kl_tensor.masked_fill(~mask, 0.0)

    # mean
    kl_sum = kl_tensor_masked.sum()
    num_valid = mask.sum()
    kl_mean = kl_sum / (num_valid + 1e-8)
    
    return kl_mean