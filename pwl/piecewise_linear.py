import torch
import torch.nn.functional as F
from torch.distributions import Distribution
from torch.distributions.utils import broadcast_all
from torch.distributions.kl import register_kl
from torch.distributions.constraints import interval, unit_interval

from splines import quadratic, linear
from spline_utils import SplineFunction

class PiecewiseLinear(Distribution):
    """
    General implementation of a distribution with piecewise linear PDF (piecewise quadratic CDF).
    Note: This class does not perform any normalization on x and y, and can produce densities that are not properly normalized.
    """

    def __init__(self, x, y, x_range=(0, 1), validate_args=None):
        """
        Piecewise linear distribution from PDF knot coordinates (x, y).

        Args:
            x (torch.Tensor): x coordinates of knots in PDF, [batch, n_dims, n_knots]
            y (torch.Tensor): y coordinates of knots in PDF, [batch, n_dims, n_knots]
            x_range (tuple, optional): Support of the distribution. Defaults to (0, 1).
        """
        if len(x.size()) == 2:
            x = x.unsqueeze(1)
        self.x, self.y = broadcast_all(x, y)
        self.x_range = x_range

        self.bin_widths = x[..., 1:] - x[..., :-1]
        self.bin_heights = y[..., 1:] - y[..., :-1]
        self.a, self.b, self.c = self._make_coefficients()
        super(PiecewiseLinear, self).__init__(batch_shape=(x.size(0),), validate_args=validate_args)

    def _make_coefficients(self):
        """
        Calculate coefficients (a, b, c) for each bin i with 
        CDF ax'^2 + bx' + c,
        PDF 2ax' + b

        Where x' = x - x_i
        """
        a = self.bin_heights / (2*self.bin_widths)
        b = self.y[..., :-1]

        # Area of trapezoid is (y_{i-1} + y_i)/2 * w_i
        bin_areas = self.bin_widths * (self.y[..., :-1] + self.y[..., 1:]) * 0.5
        c = torch.cumsum(bin_areas, -1)
        c = F.pad(c, pad=(1, 0), mode='constant', value=0.)
        return a, b, c

    def cdf(self, value):
        spline_params = (self.a, self.b, self.c[..., :-1], self.x[..., :-1])
        density = SplineFunction(self.x, self.c, quadratic, spline_params)
        return density(value)

    def icdf(self, value):
        spline_params = (self.a, self.b, self.c[..., :-1], self.x[..., :-1])
        density = SplineFunction(self.x, self.c, quadratic, spline_params)
        return density(value, inverse=True)

    def pdf(self, value):
        # NOTE cdf = ax^2 + bx + c, pdf is 2ax + b.
        spline_params = (2*self.a, self.b, self.x[..., :-1])
        density = SplineFunction(self.x, self.c, linear, spline_params)
        return density(value)

    def log_prob(self, value):
        return torch.log(self.pdf(value))

    def rsample(self, sample_shape=torch.Size([])):
        """
        Sample from piecewise linear density by inverse transform sampling:

        x = F^{-1}_X(u)
        where: u ~ U(u | 0, 1)
        """
        if len(sample_shape) == 0:
            shape = self.a.size()[:-1] + torch.Size([1])
        else:
            shape = self.a.size()[:-1] + sample_shape

        u = torch.rand(shape, device=self.c.device)
        out = self.icdf(u)

        if len(sample_shape) == 0:
            return out.squeeze(-1)
        else:
            return out

    def sample(self, sample_shape=torch.Size([])):
        with torch.no_grad():
            return self.rsample(sample_shape=sample_shape)

    def cross_entropy(self, other):
        """
        Closed form cross-entropy is available by summing over the partial cross-entropy of each bin:

        H[p, q] = - \int [p(x) log q(x)]
                = - \sum_i^{n_bins} ( \int_0^{w_i} p'_i(x') log q'_i(x') dx' )
        Where p', q' and x' are translated to 0, and w_i is the bin width

        TODO: When p.x != q.x (and as a result n_bins/w_i is different),
              redefine p and q so they have the same bins, while keeping the same density.

        Args:
            other (PiecewiseLinear): distribution q

        Returns:
            Tensor: cross-entropy between self and other.
        """
        def ce_bin(a, b, a2, b2, w):
            # - int_0^w (2a_1x + b_1)*ln(2a_2x + b_2) dx

            hwlog = torch.log(torch.clamp(2 * a2 * w + b2, min=1e-6))
            h_w = (2 * a2 * w + b2) * (a * (2 * a2 * w - b2) + 2 * a2 * b) * hwlog
            h_w -= 2 * a * torch.pow(a2 * w, 2)
            h_w += 2 * a2 * (a * b2 - 2 * a2 * b) * w 
            h_w += b2 * (a * b2 - 2 * a2 * b) * torch.log(b2)
            h = - (h_w) / (4 * torch.clamp(torch.pow(a2, 2), min=1e-6))
            return h

        if self.x.shape == other.x.shape and torch.all(torch.eq(self.x, other.x)):
            p = self
            q = other
        else:
            raise NotImplementedError("Cross-entropy with different bin widths is not yet implemented.")

        # Get partial cross-entropy for each bin and sum.
        h_total = torch.zeros(*p.x.shape[:-1]).type(p.x.type())
        for i in range(p.x.size(-1) - 1):
            h_bin = ce_bin(p.a[..., i], p.b[..., i], q.a[..., i], q.b[..., i], p.bin_widths[..., i])
            h_total += h_bin
        return h_total

    def entropy(self):
        """Calculate entropy of distribution, available in closed form
           by adding partial entropy of each individual bin:

           H[p] = - \sum_i^{n_bins} ( \int_0^{w_i} p(x') log p(x') dx' )

        Returns:
            Tensor: Entropy
        """
        def entropy_bin(a, b, w):
            # - int_0^w b log(b) dx
            h0 = -b * torch.log(b) * w

            # - int_0^w (2ax + b) ln(2ax + b) dx
            h = (4 * torch.pow(a * w, 2) + 4 * a * b * w + torch.pow(b, 2))
            h *= torch.log(torch.clamp(2 * a * w + b, min=1e-6))
            h += -2 * torch.pow(a * w, 2) - 2 * a * w * b - torch.pow(b, 2) * torch.log(b)
            # Mask a to prevent zero divisions.
            a_masked = a.masked_fill(torch.abs(a) < 1e-6, 1)
            h = -h / (4 * a_masked)

            return torch.where(a == 0, h0, h)

        h_total = torch.zeros(*self.x.shape[:-1]).type(self.x.type())
        for i in range(self.x.size(-1)-1):
            h_bin = entropy_bin(self.a[..., i], self.b[..., i], self.bin_widths[..., i])
            h_total += h_bin
        return h_total


class BinaryPWL(PiecewiseLinear):

    arg_constraints = {'p': unit_interval}
    support = unit_interval

    def __init__(self, p, w_c=0.35, w_t=0.05, h_d=0.01, s=0.1, validate_args=None):
        """
        BPWL distribution, a continuous relaxation of a binary variable with a closed form KL.

        Note that a difference here from the original implementation is that the scaling of parameter p 
        is now performed inside this class, as opposed to in sigmoid activation before the distribution.

        Example usage:
        p = torch.sigmoid(inference_net(x)) # shape [batch, latent_dims]
        q_z = BinaryPWL(p)
        z = q_z.rsample()
  
        Args:
            p (torch.Tensor): density around 1, ~p(x > 0.5)
            w_c (float, optional): Width of center bins. Defaults to 0.35.
            w_t (float, optional): Width of transition bins. Defaults to 0.05.
            h_d (float, optional): Height of center knot. Defaults to 0.01.
            s (float, optional): Slope of line segments. Defaults to 0.1.
        """
        


        # Scale p to [h_d/2, 1-h_d/2]
        self.p = p * (1-h_d) + (h_d/2)
        self.w_c = w_c
        self.w_t = w_t
        self.h_d = h_d
        self.s = s
        self.w_s = 0.5 - w_c - w_t

        self._validate_hparams()

        bin_widths = torch.Tensor([self.w_s, self.w_t, self.w_c, self.w_c, self.w_t, self.w_s])
        x = F.pad(torch.cumsum(bin_widths, dim=-1), pad=(1, 0), mode='constant', value=0.)
        x = x.view(1, 1, -1).expand(*p.shape, -1).to(p.device)

        d0 = self._get_height_norm(1-self.p)
        d1 = self._get_height_norm(self.p)

        zeros = torch.zeros_like(self.p).to(p.device)
        y = torch.stack([d0, (1-s)*d0, s*d0, zeros, s*d1, (1-s)*d1, d1], dim=-1) + h_d
        super(BinaryPWL, self).__init__(x, y, validate_args=validate_args)

    def _get_height_norm(self, p):
        return 2*(p - self.h_d*0.5) / (0.5 * (2-self.s) - 2 * self.w_c*(1-self.s) - self.w_t * (1-self.s))    
        # ((self.w_c + 2)*self.s - self.s**2 + self.w_t)

    def _validate_hparams(self):
        if not 1e-6 < self.w_c < 0.5:
            raise ValueError('w_c out of range.')
        if not 1e-6 < self.w_t < 0.5:
            raise ValueError('w_t out of range')
        if not 1e-6 < self.w_s < 0.5:
            raise ValueError('w_s out of range')

@register_kl(PiecewiseLinear, PiecewiseLinear)
def kl_pwl_pwl(p, q):
    return p.cross_entropy(q) - p.entropy()


# @register_prior_parameterization(BinaryPWL)
# def parametrize(batch_shape, event_shape, params, device, dtype):
#     if len(params) == event_shape[0]:
#         d0 = torch.Tensor(params).type(dtype)
#         d0 = d0.repeat(batch_shape + [1]).to(device)
#     elif len(params) == 1:
#         d0 = torch.full(batch_shape + event_shape, params[0],
#                         device=device, dtype=dtype)
#     else:
#         raise ValueError('invalid number of params: {}'.format(len(params)))
#     return BinaryPWL(d0)


# @register_conditional_parameterization(BinaryPWL)
# def make_binaryPWL(inputs, event_size):
#     if not inputs.size(-1) == event_size:
#         raise ValueError(
#             "Expected [...,%d] got [...,%d]" % (event_size, inputs.size(-1))
#         )

#     # Scale input between (DMIN, 1-DMIN). DMIN should be larger than middle.
#     inputs_scaled = torch.sigmoid(inputs) * (1 - 2 * DMIN) + DMIN
#     return BinaryPWL(inputs_scaled)