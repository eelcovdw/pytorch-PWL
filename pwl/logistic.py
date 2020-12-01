import torch
from torch.distributions import Distribution, Uniform
from torch.distributions.transforms import SigmoidTransform, AffineTransform
from torch.distributions import TransformedDistribution

class Logistic(TransformedDistribution):
    def __init__(self, loc, scale, validate_args=None):
        self.loc = loc
        self.scale = scale
        base_distribution = Uniform(torch.Tensor([0]).to(loc.device), torch.Tensor([1]).to(loc.device))
        transforms = [SigmoidTransform().inv, AffineTransform(loc=loc, scale=scale)]
        super().__init__(base_distribution, transforms, validate_args=validate_args)

def truncated_logistic_cdf(g, loc, scale, alpha, g0, gk):
    t1 = torch.sigmoid((g - loc)/scale)
    t2 = torch.sigmoid((g0 - alpha/2 - loc) / scale)
    t3 = torch.sigmoid((gk + alpha/2 - loc) / scale)
    return (t1 - t2) / (t3 - t2)

class DLogistic(Distribution):
    def __init__(self, loc, scale, alpha, g0, gk, validate_args=None):
        self.loc = loc
        self.scale = scale
        self.alpha = alpha
        self.g0 = g0
        self.gk = gk
        super().__init__(batch_shape=loc.size(0), event_shape=loc.size(1), validate_args=validate_args)

    def log_prob(self, x):
        return torch.log(self.pmf(x))

    def pmf(self, x):
        left = truncated_logistic_cdf(x - self.alpha/2, self.loc.unsqueeze(-1), self.scale.unsqueeze(-1), self.alpha, self.g0, self.gk)
        right = truncated_logistic_cdf(x + self.alpha/2, self.loc.unsqueeze(-1), self.scale.unsqueeze(-1), self.alpha, self.g0, self.gk)
        return right - left


if __name__ == "__main__":
    loc = torch.Tensor([[1.5]])
    scale = torch.Tensor([[1]])
    alpha = 1
    g0 = -10
    gk = 10
    d = DLogistic(loc, scale, alpha, g0, gk)
    x = torch.arange(g0, gk + 1).view(1, -1)
    print(d.pmf(x), torch.sum(d.pmf(x)))
