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
    q_log = Logistic(logits, torch.Tensor([1.]).to(logits.device))
    probs = torch.sigmoid(logits)
    with torch.no_grad():
        z_log = q_log.sample().squeeze(-1)
        u = torch.sigmoid(z_log - logits)
        
        sample_1 = ((1 - u) < probs).float()
        reward_1 = model.p_x(sample_1).log_prob(x).sum(-1)
        sample_2 = (u < probs).float()
        reward_2 = model.p_x(sample_2).log_prob(x).sum(-1)
        reward = 0.5 * (reward_1 - reward_2)
        reward = reward.unsqueeze(1) * (2*u - 1)
    surrogate = reward * logits

    loss = -(elbo + surrogate.mean(-1)).mean()

    return_dict = {
        "loss": loss,
        "kl": kl.detach(),
        "ll": ll.detach(),
        "surrogate": surrogate
    }
    return return_dict

def build_model(latent_size, latent_dist):
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
    return model

@register_conditional_parameterization(Bernoulli)
def make_bernoulli(inputs, event_size):
    assert inputs.size(-1) == event_size, "Expected [...,%d] got [...,%d]" % (event_size, inputs.size(-1))
    return Bernoulli(logits=torch.clamp(inputs, -10, 10))
