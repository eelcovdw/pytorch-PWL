import torch
from torch.utils.data import Dataset, DataLoader


def make_zemel_data(batch_size, n, p):
    """
    Creates an n*n grid with 2*n corresponding binary factors, where targets[i, j] = factors[i] | factors[j+n]

    Slight modification of the binary factor dataset used in Richard Zemels thesis [1].
    Instead of setting a preset number of factors to 1, the factors are drawn from a Bernoulli with probability p.

    [1] Zemel, Richard S. A minimum description length framework for unsupervised learning.
    """
    dist = torch.distributions.Bernoulli(probs=p)
    factors = dist.sample(sample_shape=torch.Size([batch_size, 2*n]))
    targets = torch.zeros([batch_size, n, n])
    targets[factors[:, :n] == 1] = 1
    targets = targets.transpose(1, 2)
    targets[factors[:, n:] == 1] = 1
    return factors, targets
