import torch

EPS = 1e-6

def linear(value, a, b, x0, inverse=False):
    """
    returns ax' + b or its inverse, where x' = x-x0
    """
    if inverse:
        return ((value - b) / a) + x0

    val2 = value - x0
    out = a * val2 + b
    return out


def quadratic(value, a, b, c, x0, inverse=False):
    """
    returns ax'^2 + bx' + c or its inverse, where x' = x-x0
    """

    assert all([value.size() == param.size() for param in (a, b, c, x0)])

    if not inverse:
        val2 = value - x0
        return a * torch.pow(val2, 2) + b * val2 + c

    else:
        # NOTE: torch.where breaks if either condition produces NaNs.
        # To keep division (D - b)/2a stable, set min(|a|) = EPS
        a_masked = a.masked_fill(torch.abs(a) < EPS, EPS)
        result = torch.where(
            torch.abs(a) > EPS,
            ((-b + torch.sqrt(torch.pow(b, 2) - 4*a*(c-value))) / (2*a_masked)),
            ((value - c) / b)
        )

        result = result + x0
        return result