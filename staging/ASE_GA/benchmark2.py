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
import os
import time
import torch
from amptorch.ase_utils import AMPtorch
from amptorch.descriptor.Gaussian import GaussianDescriptorSet
from amptorch.trainer import AtomsTrainer
from skorch.callbacks import EarlyStopping
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    WhiteKernel,
    ConstantKernel,
    RBF,
    DotProduct,
    RationalQuadratic,
)


def new_config(*callbacks):
    cutoff = 6.0
    cosine_cutoff_params = {"cutoff_func": "cosine"}
    gds = GaussianDescriptorSet(elements, cutoff, cosine_cutoff_params)

    low_res_g2 = (2, [0.025, 0.025], [0.0, 3.0])
    # low_res_g5 = (5, [.001, .001], [2.0, 2.0], [-1., 1.])
    low_res_g5 = (5, [0.001], [1.0], [1.0])
    low_res_elements = ["Au"]

    gds.batch_add_descriptors(*low_res_g2, important_elements=low_res_elements)
    gds.batch_add_descriptors(*low_res_g5, important_elements=low_res_elements)

    hi_res_g2 = (2, [0.025, 0.025, 0.25, 0.25], [0.0, 3.0, 0.0, 3.0])
    hi_res_g5 = (
        5,
        [
            0.001,
            0.001,
            0.001,
        ],
        [1.0, 4.0, 1.0],
        [1.0, 1.0, -1.0],
    )
    hi_res_elements = ["Pt", "Ag"]

    gds.batch_add_descriptors(*hi_res_g2, important_elements=hi_res_elements)
    gds.batch_add_descriptors(*hi_res_g5, important_elements=hi_res_elements)

    config = {
        "model": {
            "get_forces": True,
            "num_layers": 3,
            "num_nodes": 5,
            "batchnorm": False,
            "latent": True,  # LATENT NNP
        },
        "optim": {
            "force_coefficient": 0.25,
            "lr": 1e-2,
            "batch_size": 25,
            "epochs": 1000,
            "loss": "mse",
            "metric": "mae",
            "gpus": 0,
            "callbacks": list(callbacks),
        },
        "dataset": {
            "raw_data": None,
            "val_split": 0.2,
            "fp_params": gds,  # either a GDS or the `Gs` dict can be passed here
            "save_fps": True,
            # feature scaling to be used - normalize or standardize
            # normalize requires a range to be specified
            "scaling": {"type": "normalize", "range": (0, 1)},
        },
        "cmd": {
            "debug": False,
            "run_dir": "./",
            "seed": 1,
            "identifier": "test",
            "verbose": True,
            # Weights and Biases used for logging - an account(free) is required
            "logger": False,
        },
    }
    return config


def condition(training_images, *callbacks):
    config = new_config(*callbacks)
    torch.set_num_threads(1)
    trainer = AtomsTrainer(config)
    st = time.time()
    trainer.train(training_images)
    ft = time.time()
    print(
        "trainer conditioned in %0.3f seconds with a dataset of %d images"
        % (ft - st, len(training_images))
    )
    return trainer, ft - st


class ErrorPredictor(AMPtorch):
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
            "nnn": self.n_nearest_neighbors,
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

    def __call__(self, m=3):
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
        if serrors["abs"][0] > m or serrors["sqr"][0] > m:
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

    def kde_probability(self, points):
        points = np.array(points)
        reduced_x = self.dreduce(points, self.evecs[:, : self.n_components])
        lsp = self.kde(reduced_x.T)
        return 1.0 - lsp

    def latent_density(self, points, inverse=True):
        dists = cdist(points, self.latents)
        dists.sort(axis=1)
        if np.all(dists[:, 0] == 0.0):
            dists = dists[:, 1:]
        ld = np.power(dists, -1.0).sum(axis=1)
        return np.power(ld, -1.0) if inverse else ld

    def n_nearest_neighbors_inverse(self, points, n_neighbors=10, inverse=True):
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


start_time = time.time()
cluster_composition = "Pt8Ag8"
db_file = "gaxml.db"
# if not os.path.isfile(db_file):
sg = init_slab_cluster_ga(cluster_composition, db_file=db_file)

population_size = 20
population, da, comp, pairing, mutations, data = init_ga_connection(
    db_file, population_size
)


n_initial = 10
init_structure_label = "initial_structure_%d"
init_images = []
for i in range(n_initial):
    label = init_structure_label % i
    if not os.path.isfile(label + ".traj"):
        relax_atoms_emt(
            sg.get_new_candidate(),
            trajectory=label + ".traj",
            logfile="-",
            max_steps=25,
        )
    data = [label, read(label + ".traj", index=":")]
    data.append(len(data[1]))
    init_images.append(data)

images = [_ for data in init_images for _ in data[1]]
elements = np.unique([atom.symbol for atom in images[0]])
# true_energies = np.array([image.get_potential_energy() for image in images])  # [:200]])

config = new_config(EarlyStopping(patience=50, threshold=0.10))
config["dataset"]["raw_data"] = images[:1]
trainer = AtomsTrainer(config)
trainer.load_pretrained(
    "C:\\Users\\ericm\\Documents\\Research\\Code\\Amptorch\\staging\\ASE_GA\\checkpoints\\2021-02-28-00-19-59-test"
)
print("trainer loaded")

n_components = 10
kernel = WhiteKernel() + (ConstantKernel() * DotProduct())
ep = ErrorPredictor(
    trainer, images, n_components=n_components, kernel=kernel, disable_tqdm=False
)
print("ErrorPredictor made")
finish_time = time.time()
print(
    "preparation finished in %0.3f seconds, beginning interesting/interactive phase"
    % (finish_time - start_time)
)

# results = ep.predict([test_atoms])
# pred_energies, pred_forces, latents, data = results
# print(pred_energies)
# for k, v in data.items():
#     print(k, v)

# calc = AMPtorch(trainer)
# test_atoms.set_calculator(calc)

test_atoms = read("sample_start_01.traj")  # sg.get_new_candidate()
ml_traj = "ml_test3.traj"
dyn = SciPyFminBFGS(ep.connect_opt_atoms(test_atoms), trajectory=ml_traj, logfile="-")


class HighPredictedError(OptimizerConvergenceError):
    pass


dyn.attach(ep)
dyn.run(fmax=0.05, steps=100)

trajectory = read(ml_traj, index=":")
view(trajectory)
