import matplotlib.pyplot as plt
import torch
import numpy as np


import matplotlib.pyplot as plt
import torch
import numpy as np


class SplineShape:
    def __init__(self, num_knots, min_knots=3, max_knots=np.inf):
        if not min_knots <= num_knots <= max_knots or num_knots % 2 == 0:
            raise ValueError(f"Invalid number of knots: {num_knots}")
        self.min_knots = min_knots
        self.max_knots = max_knots
        self.num_knots = num_knots
        self.base_shape = self._base_shape()
    
    def _base_shape(self):
        """
        Returns a base spline shape
        """
        raise NotImplementedError()

    
    def shape(self, density, min_height):
        return self.base_shape * (density - min_height) + min_height
    

class ExponentialShape(SplineShape):
    def __init__(self, num_knots, exp_base):
        """
        A spline bump where y[n] = exp_base^n - 1 + min_height
        """
        self.exp_base = exp_base
        super().__init__(num_knots)

    def _base_shape(self):
        n = self.num_knots // 2
        s = torch.pow(self.exp_base, torch.arange(0, n+1)) - 1
        s = s / self._normalizer()
        return torch.cat([s, torch.flip(s[:-1], [0])], 0)
        
    def _normalizer(self):
        a = self.exp_base
        k = self.num_knots // 2
        n = (a**(1 + k) + k - a * (1 + k))/(a*k - k) - (a**k-1)/(2 * k)
        return n

class ZShape(SplineShape):
    def __init__(self, num_knots, slope):
        """
        A spline bump where y[n] = exp_base^n - 1 + min_height
        """
        assert num_knots == 7
        self.slope = slope
        super().__init__(num_knots=7)

    def _base_shape(self):
        left = torch.Tensor([0, self.slope, 1-self.slope, 1])
        return torch.cat([left, torch.flip(left[:-1], [0])], 0) * 2