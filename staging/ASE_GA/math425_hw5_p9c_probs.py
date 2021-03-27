import numpy as np
from scipy.special import comb


def poisson_prob(x, lam):
    return np.power(lam, x) * np.exp(-lam) / np.math.factorial(x)


def neg_bin_prob(x, r, p):
    return comb(x - 1, r - 1) * np.power(p, r) * np.power(1.0 - p, x - r)


def binomial_prob(x, n, p):
    return comb(n, x) * np.power(p, x) * np.power(1.0 - p, n - x)


probs = []
for i in range(6 + 1):
    row = []
    row.append(poisson_prob(i, lam=3))
    row.append(neg_bin_prob(i, r=3, p=0.7))
    row.append(binomial_prob(i, n=10, p=0.7))
    probs.append(row)
probs = np.array(probs).T
print(np.round(probs, 4))
cond_probs = np.array([(p[6] + p[5]) / p.sum() for p in probs])
print("P_Poisson (X > 4 | X <= 6) = %f" % cond_probs[0])
print("P_Neg_Bin (X > 4 | X <= 6) = %f" % cond_probs[1])
print("P_Binom   (X > 4 | X <= 6) = %f" % cond_probs[2])
