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
