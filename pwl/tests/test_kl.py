import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from piecewise_linear import BinaryPWL

plt.style.use("seaborn")

def sample_between(min_val, max_val):
    """sample random float between min_val and max_val"""
    return np.random.random() * (max_val - min_val) + min_val

def estimate_CE(p, q, n_samples=1):
    """MC estimate of H(p, q), using n_samples"""
    x = p.sample(sample_shape=torch.Size([n_samples]))
    cross_entropy = - q.log_prob(x)
    return cross_entropy

def estimate_entropy(p, n_samples=1):
    """MC estimate of entropy H(p), using n_samples."""
    x = p.sample(sample_shape=torch.Size([n_samples]))
    p_entropy = - p.log_prob(x)
    return p_entropy

def estimate_kl(p, q, n_samples=1):
    """MC estimate of KL(p||q), using n samples."""
    x = p.sample(sample_shape=torch.Size([n_samples]))
    p_entropy = - p.log_prob(x)
    cross_entropy = - q.log_prob(x)
    return cross_entropy - p_entropy

def get_random_bpwl_params(rng_h_d=[0.01, 0.1], rng_s=[0.01, 0.2]):
    """Returns random bpwl hyperparams [w_c, w_t, w_s],  h_d, s"""
    bin_widths = np.random.random(3)
    bin_widths = (bin_widths+0.01) / (2*np.sum(bin_widths+0.01))
    h_d = sample_between(*rng_h_d)
    s = sample_between(*rng_s)
    return bin_widths, h_d, s

if __name__=="__main__":
    print("This script compares the closed-form entropy, cross entropy and KL-divergence of a random BPWL against MC-estimates.")

    batch_size = 8
    n_dims = 4

    # Random hparams
    bin_widths, h_d, s = get_random_bpwl_params()
    n_samples = [int(i) for i in np.linspace(1, 10000, 500)]
    print("Distribution hyperparameters")
    print("w_c/w_t/w_s:\t{:.3f}/{:.3f}/{:.3f}".format(*bin_widths))
    print("h_d:\t{:.3f}".format(h_d))
    print("s:\t{:.3f}".format(s))

    # Make distributions
    p1 = torch.rand(batch_size, n_dims) 
    dist1 = BinaryPWL(p1, bin_widths[0], bin_widths[1], h_d, s, validate_args=True)

    p2 = torch.rand(batch_size, n_dims)
    dist2 = BinaryPWL(p2, bin_widths[0], bin_widths[1], h_d, s, validate_args=True)

    # Estimate
    print("Estimating Entropy...")
    ent = dist1.entropy().sum(-1).mean()
    ent_estimates = []
    for n in tqdm(n_samples):
        ent_estimates.append(estimate_entropy(dist1, n).sum(1).mean())

    print("Estimating Cross-entropy...")
    xent = dist1.cross_entropy(dist2).sum(-1).mean()
    xent_estimates = []
    for n in tqdm(n_samples):
        xent_estimates.append(estimate_CE(dist1, dist2, n).sum(1).mean())

    print("Estimating KL...")
    kl = torch.distributions.kl_divergence(dist1, dist2).sum(-1).mean()
    kl_estimates = []
    for n in tqdm(n_samples):
        kl_estimates.append(estimate_kl(dist1, dist2, n).sum(1).mean())

    # Plot results
    plt.figure(figsize=[15, 5])
    plt.subplot(1,3,1)
    plt.plot(n_samples, ent_estimates, label="Estimate")
    plt.plot(n_samples, [ent.item()]*len(n_samples), label="Closed-form")
    plt.title("Entropy")
    plt.xlabel("Samples")
    plt.legend()

    plt.subplot(1,3,2)
    plt.plot(n_samples, xent_estimates, label="Estimate")
    plt.plot(n_samples, [xent.item()]*len(n_samples), label="Closed-form")
    plt.title("Cross-entropy")
    plt.xlabel("Samples")
    plt.legend()

    plt.subplot(1,3,3)
    plt.plot(n_samples, kl_estimates, label="Estimate")
    plt.plot(n_samples, [kl.item()]*len(n_samples), label="Closed-form")
    plt.title("KL")
    plt.xlabel("Samples")
    plt.legend()
    plt.show()
