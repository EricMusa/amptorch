from ase import db
from staging.ASE_GA.ga_initialize import init_slab_cluster_ga
from staging.ASE_GA.ga_operations import (
    init_ga_connection,
    relax_atoms_emt,
    relax_unrelaxed_candidates,
    generate_relaxed_candidate,
)
import numpy as np
from scipy.stats import gaussian_kde, maxwell
from scipy.spatial.distance import cdist, pdist, squareform
import scipy.linalg as la
from ase.io import read
from ase.optimize.sciopt import SciPyFminBFGS, Converged, OptimizerConvergenceError
from ase.visualize import view
from ase.calculators.emt import EMT
import numpy as np
from amptorch.ase_utils import AMPtorch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    WhiteKernel,
    ConstantKernel,
    DotProduct,
)

class Predictor:
    def __init__(
        self, trainer, dataset, n_components=10, kernel=None, disable_tqdm=True
    ):
        self.trainer = trainer
        self.dataset = dataset
        self.true_energies = np.array(
            [atoms.get_potential_energy() for atoms in self.dataset]
        )
        self.opt_atoms = None

        print("making predictions")
        self.predictions = self.trainer.predict(self.dataset, disable_tqdm=disable_tqdm)
        self.latents = np.array(self.predictions["latent"])
        self.latent_dims = self.latents.shape[1]
        self.pred_energies = np.array(self.predictions["energy"])
        self.pred_forces = np.array(self.predictions["forces"])

        print("evaluating principality of latent data")
        self.n_components = n_components
        self.evecs, self.evals = self.pca(self.latents)
        print(
            "principal components calculated, reducing data to %d components"
            % self.n_components
        )
        self.reduced_data = self.dreduce(
            self.latents, self.evecs[:, : self.n_components]
        )

        self.metrics = {}
        self.metric_funcs = {
            # "nnn": self.n_nearest_neighbors,
            "kde": self.kde_probability,
            "inv": self.latent_density,
            "nni": self.n_nearest_neighbors_inverse,
        }
        print("available metrics:", list(self.metric_funcs.keys()))
        self.n_neighbors = 10

        self.kde = gaussian_kde(self.reduced_data.T)

        self.errors = {}
        self.error_funcs = {
            "abs": self.absolute_errors,
            "sqr": self.square_errors,
        }
        print("available errors:", list(self.error_funcs.keys()))
        self.stats = {}
        kernel = kernel or (WhiteKernel() + ConstantKernel() * DotProduct())
        self.error_gpr = GaussianProcessRegressor(kernel)
        self.condition_error_gpr()

    def condition_error_gpr(self):
        print("calculating latent space metrics")
        self.metrics = self.calculate_metrics(self.latents)
        for name in self.metric_funcs.keys():
            self.stats[name] = [self.metrics[name].mean(), self.metrics[name].std()]

        print("calculating errors")
        self.errors = self.calculate_errors()
        for name in self.error_funcs.keys():
            self.stats[name] = [self.errors[name].mean(), self.errors[name].std()]

        print("fitting error GPR to standardized metrics and errors")
        standard_metrics = np.vstack(
            [self.standardize(v, k) for k, v in self.metrics.items()]
        ).T
        standard_errors = np.vstack(
            [self.standardize(v, k) for k, v in self.errors.items()]
        ).T
        self.error_gpr.fit(standard_metrics, standard_errors)
        print("ErrorPredictor ready!")

    def predict(self, atoms, disable_tqdm=True):
        predictions = self.trainer.predict(atoms, disable_tqdm=disable_tqdm)
        energies = np.array(predictions["energy"])
        forces = np.array(predictions["forces"])
        latents = np.array(predictions["latent"])
        metrics = self.calculate_metrics(latents)
        smetrics = {k: self.standardize(v, k) for k, v in metrics.items()}
        serrors = self.error_gpr.predict(np.vstack(list(smetrics.values())).T)
        serrors = {k: serrors[:, i] for i, k in enumerate(self.errors.keys())}
        errors = {k: self.destandardize(v, k) for k, v in serrors.items()}
        data = {
            "metrics": metrics,
            "smetrics": smetrics,
            "serrors": serrors,
            "errors": errors,
        }
        return energies, forces, latents, data

    def connect_opt_atoms(self, opt_atoms):
        self.opt_atoms = opt_atoms
        self.opt_atoms.set_calculator(AMPtorch(self.trainer))
        return opt_atoms

    def __call__(self, m=3.):
        self.opt_atoms.wrap()
        energy = self.opt_atoms.get_potential_energy()
        forces = self.opt_atoms.get_forces()
        latents = self.opt_atoms.info["latent"]
        metrics = self.calculate_metrics(latents)
        smetrics = {k: self.standardize(v, k) for k, v in metrics.items()}
        serrors = self.error_gpr.predict(np.vstack(list(smetrics.values())).T)
        serrors = {k: serrors[:, i] for i, k in enumerate(self.errors.keys())}
        errors = {k: self.destandardize(v, k) for k, v in serrors.items()}
        data = {
            "metrics": metrics,
            "smetrics": smetrics,
            "errors": errors,
            "serrors": serrors,
        }
        # atoms = self.opt_atoms.copy()
        # atoms.set_calculator(EMT())
        # true_energy = atoms.get_potential_energy()
        # true_errors = np.array([np.abs(energy - true_energy), np.power(energy - true_energy, 2.)])
        print("energy:", np.round(energy, 4))
        print("predicted errors:", errors["abs"][0], errors["sqr"][0])
        print("standard errors:", serrors["abs"][0], serrors["sqr"][0])
        if np.array([serrors["abs"][0], serrors["sqr"][0]]).mean() > m:
            vals = (
                errors["abs"][0],
                errors["sqr"][0],
                serrors["abs"][0],
                serrors["sqr"][0],
            )
            raise HighPredictedError(
                "Predicted errors (Abs=%0.3f eV, Sqr=%0.3f eV^2) are (%0.3f, %0.3f) standard deviations from the means"
                % vals
            )
        # print('actual errors:', true_errors)
        # for k, v in data.items():
        #     print('%s:' % k, v)

    def calculate_metrics(self, latent_points):
        return {
            name: metric_func(latent_points.reshape(-1, self.latent_dims))
            for name, metric_func in self.metric_funcs.items()
        }

    def calculate_errors(self):
        return {name: error_func() for name, error_func in self.error_funcs.items()}

    def standardize(self, values, name):
        return (values - self.stats[name][0]) / self.stats[name][1]

    def destandardize(self, values, name):
        return values * self.stats[name][1] + self.stats[name][0]

    def n_nearest_neighbors(
        self,
        points,
        n_neighbors=10,
    ):
        n_neighbors = n_neighbors or self.n_neighbors
        dists = cdist(points, self.latents)
        dists.sort(axis=1)
        if np.all(dists[:, 0] == 0.0):
            dists = dists[:, 1:]
        return dists[:, :n_neighbors].sum(axis=1)

    def kde_probability(self, points, inverse=False):
        points = np.array(points)
        reduced_x = self.dreduce(points, self.evecs[:, : self.n_components])
        lsp = self.kde(reduced_x.T)
        return 1. - lsp if inverse else lsp

    def latent_density(self, points, inverse=False):
        dists = cdist(points, self.latents)
        dists.sort(axis=1)
        if np.all(dists[:, 0] == 0.0):
            dists = dists[:, 1:]
        ld = np.power(dists, -1.0).sum(axis=1)
        return np.power(ld, -1.0) if inverse else ld

    def n_nearest_neighbors_inverse(self, points, n_neighbors=10, inverse=False):
        n_neighbors = n_neighbors or self.n_neighbors
        dists = cdist(points, self.latents)
        dists.sort(axis=1)
        if np.all(dists[:, 0] == 0.0):
            dists = dists[:, 1:]
        nnni = np.power(dists[:, :n_neighbors], -1.0).sum(axis=1)
        return np.power(nnni, -1.0) if inverse else nnni

    def absolute_errors(
        self,
    ):
        return np.abs(self.pred_energies - self.true_energies)

    def square_errors(
        self,
    ):
        return np.power(self.pred_energies - self.true_energies, 2.0)

    @staticmethod
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

    @staticmethod
    def dreduce(data, primary_component_vectors):
        return np.dot(primary_component_vectors.T, data.T).T


class HighPredictedError(OptimizerConvergenceError):
    pass
