import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.distributions import kl_divergence, Normal, Bernoulli, RelaxedBernoulli
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from probabll.dgm import register_conditional_parameterization
from probabll.dgm.prior import PriorLayer
from probabll.dgm.conditional import ConditionalLayer
from probabll.dgm.conditioners import FFConditioner
from probabll.dgm.likelihood import FullyFactorizedLikelihood

import sys
sys.path.append('./..')
import binary_concrete
from logistic import Logistic
from piecewise_linear import BinaryPWL, BinaryPWL2

class VAE(torch.nn.Module):
    def __init__(self, x_size, z_size, conditional_x, conditional_z, prior_z):
        super().__init__()
        self.x_size = x_size
        self.z_size = z_size
        self.conditional_z = conditional_z
        self.conditional_x = conditional_x
        self.prior_z = prior_z

    def inference_parameters(self):
        return self.conditional_z.parameters()

    def generative_parameters(self):
        return self.conditional_x.parameters()

    def q_z(self, x):
        return self.conditional_z(x)

    def p_z(self, batch_size, device):
        return self.prior_z(batch_size, device)

    def p_x(self, z):
        return self.conditional_x(z)


def train_step_grep(model, x):
    # Forward through model
    p_z = model.p_z(x.size(0), x.device)
    q_z = model.q_z(x)
    z = q_z.rsample()
    p_x = model.p_x(z)

    # Loss
    ll = p_x.log_prob(x).sum(-1)
    kl = kl_divergence(q_z, p_z).sum(-1)
    elbo = ll - kl
    loss = -elbo.mean()

    return_dict = {
        "loss": loss,
        "kl": kl.detach(),
        "ll": ll.detach()
    }

    return return_dict


def train_step_sfe(model, x):
    """
    SFE is implemented for reference, using ARM is recommended.
    """
    p_z = model.p_z(x.size(0), x.device)
    q_z = model.q_z(x)
    z = q_z.sample()
    p_x = model.p_x(z)

    ll = p_x.log_prob(x).sum(-1)
    kl = kl_divergence(q_z, p_z).sum(-1)    
    elbo = ll - kl
    
    # gradients w.r.t q_z with SFE
    surrogate = ll.detach() * q_z.log_prob(z).sum(-1)

    loss = -(elbo + surrogate).mean()

    return_dict = {
        "loss": loss,
        "kl": kl.detach(),
        "ll": ll.detach(),
        "surrogate": surrogate.detach()
    }
    return return_dict

def train_step_arm(model, x):
    p_z = model.p_z(x.size(0), x.device)
    q_z = model.q_z(x)
    z = q_z.sample()
    p_x = model.p_x(z)

    ll = p_x.log_prob(x).sum(-1)
    kl = kl_divergence(q_z, p_z).sum(-1)
    elbo = ll - kl
    
    # gradients w.r.t q_z with ARM Estimator
    logits = q_z.logits
    q_log = Logistic(logits, torch.Tensor([1.]).to(logits.device))
    with torch.no_grad():
        eps = Logistic(torch.Tensor([0]), torch.Tensor([1])).sample(sample_shape=z.size()).to(logits.device).squeeze(-1)
        sample_1 = logits + eps
        reward_1 = model.p_x((sample_1 > 0.).float()).log_prob(x).sum(-1)
        sample_2 = logits - eps
        reward_2 = model.p_x((sample_2 > 0.).float()).log_prob(x).sum(-1)
        reward = (reward_1 - reward_2) / 2
    surrogate = reward * q_log.log_prob(sample_1).sum(-1)

    loss = -(elbo + surrogate).mean()

    return_dict = {
        "loss": loss,
        "kl": kl.detach(),
        "ll": ll.detach(),
        "surrogate": surrogate.detach()
    }
    return return_dict


def train_step_disarm(model, x):
    p_z = model.p_z(x.size(0), x.device)
    q_z = model.q_z(x)
    z = q_z.sample()
    p_x = model.p_x(z)

    ll = p_x.log_prob(x).sum(-1)
    kl = kl_divergence(q_z, p_z).sum(-1)
    elbo = ll - kl
    
    # gradients w.r.t q_z with ARM
    logits = q_z.logits
    f = lambda b: model.p_x(b).log_prob(x).sum(-1)
    with torch.no_grad():
        u = torch.rand(logits.size()).to(logits.device)
        eps = torch.logit(u)
        b = (logits + eps > 0.).float()
        b_ = (logits - eps > 0.).float()
        r = f(b)
        r_ = f(b_)
        s = torch.pow(-1, b_) * (~torch.eq(b, b_)).float() * torch.sigmoid(torch.abs(logits))
        reward = 0.5 * (r - r_).unsqueeze(-1) * s
    surrogate = (reward * logits).sum(-1)

    loss = -(elbo + surrogate).mean()

    return_dict = {
        "loss": loss,
        "kl": kl.detach(),
        "ll": ll.detach(),
        "surrogate": surrogate
    }
    return return_dict

def train_step_arm_2(model, x):
    p_z = model.p_z(x.size(0), x.device)
    q_z = model.q_z(x)
    z = q_z.sample()
    p_x = model.p_x(z)

    ll = p_x.log_prob(x).sum(-1)
    kl = kl_divergence(q_z, p_z).sum(-1)
    elbo = ll - kl
    
    # gradients w.r.t q_z with ARM
    logits = q_z.logits
    f = lambda b: model.p_x(b).log_prob(x).sum(-1)
    with torch.no_grad():
        u = torch.rand(logits.size()).to(logits.device)
        eps = torch.logit(u)
        r = f((logits + eps > 0.).float())
        r_ = f((logits - eps > 0.).float())
        reward = 0.5 * (r - r_).unsqueeze(-1) * (2 * u - 1)
    surrogate = (reward * logits).sum(-1)

    loss = -(elbo + surrogate).mean()

    return_dict = {
        "loss": loss,
        "kl": kl.detach(),
        "ll": ll.detach(),
        "surrogate": surrogate
    }
    return return_dict

def build_model(latent_size, latent_dist, estimator):
    if latent_dist == "bernoulli":
        posterior_type = Bernoulli
        prior_type = Bernoulli
        inf_output_size = latent_size
        prior_params = [0.5]
    elif latent_dist == "normal":
        posterior_type = Normal
        prior_type = Normal
        inf_output_size = latent_size * 2
        prior_params = [0., 1.]
    elif latent_dist == "bpwl":
        posterior_type = BinaryPWL
        prior_type = BinaryPWL
        inf_output_size = latent_size
        prior_params = [0.5]
    elif latent_dist == "bpwl2":
        posterior_type = BinaryPWL2
        prior_type = BinaryPWL2
        inf_output_size = latent_size
        prior_params = [0.5]
    elif latent_dist == "binary-concrete":
        posterior_type = RelaxedBernoulli
        prior_type = RelaxedBernoulli
        inf_output_size = latent_size
        prior_params = [0.5]
    else:
        raise ValueError(f'Unknown latent distribution: {latent_dist}')

    if estimator.lower() == "sfe":
        train_step = train_step_sfe
    elif estimator.lower() == "grep":
        train_step = train_step_grep
    elif estimator.lower() == "arm" and latent_dist == "bernoulli":
        train_step = train_step_arm_2
    elif estimator.lower() == "disarm" and latent_dist == "bernoulli":
        train_step = train_step_disarm
    else:
        raise ValueError(f"Unknown latent/estimator combination: {latent_dist} / {estimator}")

    conditional_z = ConditionalLayer(
        event_size=latent_size,
        dist_type=posterior_type,
        conditioner=FFConditioner(
            input_size=28*28,
            output_size=inf_output_size,
            hidden_sizes=[200, 200],
            hidden_activation=torch.nn.LeakyReLU()
        )
    )
    prior_z = PriorLayer(
        event_shape=latent_size,
        dist_type=prior_type,
        params=prior_params
    )
    conditional_x = FullyFactorizedLikelihood(
        event_size=28*28,
        dist_type=Bernoulli,
        conditioner=FFConditioner(
            input_size=latent_size,
            output_size=28*28,
            hidden_sizes=[200, 200],
            hidden_activation=torch.nn.LeakyReLU()
        )
    )

    model = VAE(28*28, latent_size, conditional_x, conditional_z, prior_z)
    return model, train_step

@register_conditional_parameterization(Bernoulli)
def make_bernoulli(inputs, event_size):
    assert inputs.size(-1) == event_size, "Expected [...,%d] got [...,%d]" % (event_size, inputs.size(-1))
    return Bernoulli(logits=torch.clamp(inputs, -10, 10))
