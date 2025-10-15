import torch
from torch.distributions import kl_divergence
from torch.distributions import Distribution, Weibull, Gamma, register_kl


@register_kl(Weibull, Gamma)
def kl_weibull_gamma(
    approx: Weibull, 
    prior: Gamma,
):
    """
    - Analytic KL divergence: KL(Weibull(k, λ) || Gamma(α, β))
    - Reference: Fan et al., "Bayesian Attention Modules", NeurIPS 2020.
    """
    k, lam = approx.concentration, approx.scale
    alpha, beta = prior.concentration, prior.rate

    gamma_const = -torch.special.digamma(torch.tensor(1.0, device=k.device))
    
    eps = 1e-8
    term1 = (gamma_const / k) - torch.log(lam * k + eps)
    term2 = torch.log(beta)
    term3 = ((alpha - 1) / k) * (gamma_const + torch.log(lam + eps))
    term4 = beta * lam * torch.exp(torch.lgamma(1.0 + 1.0 / k))
    kl = term1 + term2 + term3 + term4 - alpha
    kl = torch.clamp(kl, min=0.0)
    
    return kl

def masked_kl_fn(
    approx: torch.distributions.Distribution, 
    prior: torch.distributions.Distribution, 
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