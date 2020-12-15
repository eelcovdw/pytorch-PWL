import torch
from torch.distributions import RelaxedBernoulli
from torch.distributions.kl import register_kl
from probabll.dgm import register_conditional_parameterization, register_prior_parameterization

"""
Pytorch implements Binary Concrete in RelaxedBernoulli as a transformation of LogitRelaxedBernoulli,
which is what [1] recommends. This script only implements the KL divergence between two RelaxedBernoulli distributions.

[1] Maddison, Chris J., Andriy Mnih, and Yee Whye Teh. "The concrete distribution:
    A continuous relaxation of discrete random variables."
"""

def kl_concrete_concrete(p, q, n_samples=1):
    """
    KL is estimated for the logits of the binary concrete distribution to avoid underflow.
    """
    x_logit = p.base_dist.rsample(torch.Size([n_samples]))
    return (p.base_dist.log_prob(x_logit) - q.base_dist.log_prob(x_logit)).mean(0)

@register_kl(RelaxedBernoulli, RelaxedBernoulli)
def _kl_concrete_concrete(p, q):
    return kl_concrete_concrete(p, q, n_samples=10)


@register_prior_parameterization(RelaxedBernoulli)
def parametrize(batch_shape, event_shape, params, device, dtype):
    if len(params) == 1:
        p = torch.full(batch_shape + event_shape, params[0], device=device, dtype=dtype)
    elif len(params) == event_shape[0]:
        p = torch.Tensor(params).type(dtype).repeat(batch_shape + [1]).to(device)
    # Prior temperature from [1]
    temp = torch.Tensor([0.5]).to(device)
    return RelaxedBernoulli(temp, probs=p)


@register_conditional_parameterization(RelaxedBernoulli)
def make_relaxed_bernoulli(inputs, event_size):
    if not inputs.size(-1) == event_size:
        raise ValueError(
            "Expected [...,%d] got [...,%d]" % (event_size, inputs.size(-1))
        )
    # Posterior temperature from [1]
    temp = torch.Tensor([0.66]).to(inputs.device)
    return RelaxedBernoulli(temp, logits=inputs)
