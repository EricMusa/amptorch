import numpy as np
from scipy.optimize import curve_fit


def max_linear(x, b, u):
    # x, b, u = np.array(x), np.array(b), np.array(u)
    prod = np.dot(x, b)
    return np.where(prod > u, prod, u)


def max_linear_deriv(x, b, u):
    if max_linear(x, b, u) > u:
        return b
    else:
        return 0.








