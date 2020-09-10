import torch
from torch import nn


class SplineFunction(nn.Module):
    """
    Generic module to model a piecewise function.
    """

    def __init__(self, x, y, spline_fn, spline_params, has_inverse=True):
        """
        Args:
            x (Tensor): x coordinates of bins, including left and right bounds.
            y (Tensor): y coordinates of bins, including top and bottom bounds.
            spline_fn: spline function, applied elementwise.
                       parameters: (value, *spline_params, inverse)
            spline_params (tuple):  see spline_fn.
            has_inverse (bool): set to false if spline_fn does not have an inverse.
        """
        super(SplineFunction, self).__init__()
        self.x = x
        self.y = y
        self.spline_fn = spline_fn
        self.spline_params = spline_params
        self.has_inverse = has_inverse

    def _get_bins(self, value, inverse=False):
        """ Determine bin for each element in value.

        Args:
            value (Tensor): values to bin
            inverse (bool, optional): if inverse, use self.y to determine bins,
                                      Else use self.x. Defaults to False.

        Returns:
            Torch.LongTensor: Tensor with same shape as value containing bins.
        """
        if inverse:
            return get_bins(value, self.y)
        else:
            return get_bins(value, self.x)

    def forward(self, value, inverse=False):
        """Apply self.spline_fn to value, using bins from self.y if inverse else self.x.

        Args:
            value (Tensor): shape [batch_size, n_dims, n_samples] or [batch_size, n_dims]
            inverse (bool, optional): Apply inverse if True. Defaults to False.

        Returns:
            torch.Tensor: Tensor containing values transformed by spline function.
        """
        if inverse and not self.has_inverse:
            raise ValueError('Attempting to invert density without inverse.')

        if value.dim() == 2:
            squeeze = True
            value = value.unsqueeze(-1)
        else:
            squeeze=False

        bins = self._get_bins(value, inverse=inverse)
        bins_exp = bins.unsqueeze(-2)
        # Expand params, select correct param from each bin
        params_exp = [_share_between_samples(param, value.size(-1))
                      for param in self.spline_params]
        params_bin = [torch.gather(param, -2, bins_exp).squeeze(-2)
                      for param in params_exp]

        result = self.spline_fn(value, *params_bin, inverse=inverse)

        if squeeze:
            result = result.squeeze(-1)
        return result


def _share_between_batch(params, batch_size):
    """ Share parameter across n samples: [*dims] -> [batch_size, *dims]."""
    return params.unsqueeze(0).expand(batch_size, *params.shape)


def _share_between_samples(params, n_samples):
    """ Share parameter across n samples: [*dims] -> [*dims, n_samples]"""
    return params.unsqueeze(-1).expand(*params.shape, n_samples)


def get_bins(value, bin_locations):
    """Determine bin for each element in value.

    Args:
        value (Tensor): values to bin, shape [batch_size, n_dims, n_samples]
        bin_locations (Tensor): shape [batch_size, n_dims, n_bins+1]

    Returns:
        LongTensor
    """
    x_e = _share_between_samples(bin_locations, value.size(-1))
    bins = torch.zeros_like(value)
    # For each bin, set bins to i if value in bin.
    for i in range(bin_locations.size(-1)-1):
        bin_mask = (x_e[:, :, i] <= value) * (value <= x_e[:, :, i+1])
        bins[bin_mask] = i

    return bins.type(torch.LongTensor).to(value.device)
