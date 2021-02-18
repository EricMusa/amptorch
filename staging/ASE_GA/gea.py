import numpy as np
from scipy.stats import gaussian_kde, maxwell
from scipy.spatial.distance import pdist, squareform
import scipy.linalg as la


def pca(data):
    """
    adapted from:
    https://stackoverflow.com/questions/13224362/principal-component-analysis-pca-in-python
    https://stackoverflow.com/users/66549/doug

    data: (n x d)
    """
    data = np.array(data)
    data -= data.mean(axis=0)
    cov = np.cov(data, rowvar=False)
    eigvals, eigvecs = la.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    eigvals = eigvals[idx]
    return eigvecs, eigvals


def dreduce(data, primary_component_vectors):
    """"""
    return np.dot(primary_component_vectors.T, data.T).T


def inverse_distance_density(data):
    dmat = squareform(pdist(data))
    inv_dmat = np.power((dmat + np.diag([np.inf for _ in range(len(dmat))])), -1.)
    point_densities = inv_dmat.sum(axis=0) / (len(inv_dmat) - 1.)
    density = point_densities.sum() / 2.
    return density, point_densities, inv_dmat, dmat


class PCKDE:
    def __init__(self, data, n_components=10):
        self.data = data
        self.n_components = n_components
        print("calculating PCs")
        self.evecs, self.evals = pca(self.data)
        self.evecs = self.evecs[:, :n_components]
        print("PCs calculated, reducing data")
        self.reduced_data = dreduce(self.data, self.evecs)
        self.kde = gaussian_kde(self.reduced_data.T)

    def __call__(self, x):
        x = np.array(x)
        reduced_x = dreduce(x, self.evecs)
        return self.kde(reduced_x.T)

    def density(self, data=None, use_pcs=True):
        if data is None and use_pcs:
            data = self.reduced_data
        elif data is None and not use_pcs:
            data = self.data
        elif data is not None and use_pcs:
            data = dreduce(data, self.evecs)
        return inverse_distance_density(data)



