import numpy as np
import os
import torch
from amptorch.ase_utils import AMPtorch
from amptorch.descriptor.Gaussian import GaussianDescriptorSet
from amptorch.trainer import AtomsTrainer
from ase.io import read
from scipy.stats import gaussian_kde
from scipy.spatial.distance import pdist, squareform
import scipy.linalg as la
from itertools import combinations


class Dist:
    def __init__(self, obs):
        self.obs = obs
        self.mean = obs.mean(axis=0)
        self.std = obs.std(axis=0)
        self.std[self.std == 0.0] = 1.0

    def standard_center_distance(self, x):
        return np.linalg.norm((x - self.mean) / self.std, axis=1)

    def center_distance(self, x):
        return np.linalg.norm((x - self.mean), axis=1)

    def closest_distance(self, x):
        all_dists = cdist(self.obs, x)
        return all_dists[all_dists.argmin(axis=0), np.arange(len(x))]

    def standard_closest_distance(self, x):
        all_dists = cdist(self.obs, x, "seuclidean")
        return all_dists[all_dists.argmin(axis=0), np.arange(len(x))]

    def weighted_distance(self, x, center_weight=0.5):
        return center_weight * self.center_distance(x) + (
            1.0 - center_weight
        ) * self.closest_distance(x)

    def standard_weighted_distance(self, x, center_weight=0.5):
        return center_weight * self.standard_center_distance(x) + (
            1.0 - center_weight
        ) * self.standard_closest_distance(x)

    def agg_dist(x):
        return (
            self.center_distance(x),
            self.standard_center_distance(x),
            self.closest_distance(x),
            self.standard_closest_distance(x),
        )


def pca(data, n_components=3):
    """
    adapted from
    https://stackoverflow.com/questions/13224362/principal-component-analysis-pca-in-python
    https://stackoverflow.com/users/66549/doug
    """
    d = data.copy()
    d -= d.mean(axis=0)
    cov = np.cov(d, rowvar=False)
    evals, evecs = la.eigh(cov)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    evals = evals[idx]
    evecs = evecs[:, :n_components]
    return evecs, evals


def dreduce(data, primary_component_vectors):
    return np.dot(primary_component_vectors.T, data.T).T


class PCKDE:
    def __init__(self, data, n_components=10):
        self.data = data
        self.n_components = n_components
        print("calculating PCs")
        self.evecs, self.evals = pca(self.data, self.n_components)
        print("PCs calculated, reducing data")
        self.reduced_data = dreduce(self.data, self.evecs)
        self.kde = gaussian_kde(self.reduced_data.T)

    def __call__(self, x):
        reduced_x = dreduce(x, self.evecs)
        return self.kde(reduced_x.T)


relaxations = []
n_images = []
for fname in os.listdir("cu8_on_zno"):
    if fname.endswith(".traj"):
        print("loading %s" % fname)
        images = read(os.path.join("cu8_on_zno", fname), index=":")
        relaxations.append([fname.replace(".traj", ""), images])
        n_images.append(len(images))
        print("%s loaded, %d images" % (fname, len(relaxations[-1])))
n_images = np.array(n_images)
print(
    "total images, total relaxations:",
    sum(len(v[1]) for v in relaxations),
    len(relaxations),
)
print(
    "mean, std. dev. #images/relaxation: %.04f, %.04f"
    % (n_images.mean(), n_images.std())
)
print(n_images)

rejects = {}
for fname in os.listdir("cu8_on_zno/rejects"):
    if fname.endswith(".traj"):
        print("loading %s" % fname)
        rejects[fname.replace(".traj", "")] = read(
            os.path.join("cu8_on_zno/rejects", fname), index=":"
        )
        print(
            "%s loaded, %d images" % (fname, len(rejects[fname.replace(".traj", "")]))
        )

print("total reject images:", sum(len(v) for v in rejects.values()))
reject_images = [_ for rel in rejects.values() for _ in rel]
regular_images = [_ for rel in relaxations for _ in rel[1]]
all_images = regular_images + reject_images
seed = 123
np.random.seed(seed)
n_images = 32
images = [
    all_images[i]
    for i in np.random.choice(list(range(len(all_images))), n_images, replace=False)
]
print("using %d training images total, good + rejected relaxations" % len(images))

elements = np.unique([atom.symbol for atom in images[0]])
cutoff = 6.0
cosine_cutoff_params = {"cutoff_func": "cosine"}

gds = GaussianDescriptorSet(elements, cutoff, cosine_cutoff_params)

# low_res_g2 = (2, [.025, .025], [0., 3.])
low_res_g2 = (
    2,
    [
        0.025,
    ],
    [
        3.0,
    ],
)
# low_res_g5 = (5, [.001, .001], [2.0, 2.0], [-1., 1.])
low_res_g5 = (5, [0.001], [1.0], [1.0])
low_res_elements = ["O", "Zn"]

gds.batch_add_descriptors(*low_res_g2, important_elements=low_res_elements)
# gds.batch_add_descriptors(*low_res_g5, important_elements=low_res_elements)


hi_res_g2 = (
    2,
    [0.025, 0.25],
    [
        3.0,
        3.0,
    ],
)
hi_res_g5 = (
    5,
    [
        0.001,
        0.001,
    ],
    [1.0, 1.0],
    [1.0, -1.0],
)
# hi_res_g5 = (5, [.001,], [2.,], [1.])
hi_res_elements = ["Cu"]

gds.batch_add_descriptors(*hi_res_g2, important_elements=hi_res_elements)
gds.batch_add_descriptors(*hi_res_g5, important_elements=hi_res_elements)

print("constructed GDS hash:", gds.descriptor_setup_hash)

config = {
    "model": {
        "get_forces": True,
        "num_layers": 2,
        "num_nodes": 10,
        "batchnorm": False,
        "latent": True,
    },
    "optim": {
        "force_coefficient": 0.25,
        "lr": 1e-2,
        "batch_size": 32,
        "epochs": 10,
        "loss": "mse",
        "metric": "mae",
        "gpus": 0,
    },
    "dataset": {
        "raw_data": images,
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

print("Superficially training Amptorch model")
torch.set_num_threads(1)
trainer = AtomsTrainer(config)
trainer.train()  # interuptable without crashing
# print("Amptorch model superficially trained. Calculating latency data")
print("Superficual training complete - predicting properties of images")
predictions = trainer.predict(images, disable_tqdm=False)
print("predictions calculated")

# true_energies = np.array([image.get_potential_energy() for image in images])
# pred_energies = np.array(predictions["energy"])

# print("Energy MSE:", np.mean((true_energies - pred_energies) ** 2), "eV")
# print("Energy MAE:", np.mean(np.abs(true_energies - pred_energies)), "eV")

latents = np.array(predictions["latent"])


# levecs, levals = pca(latents, n_components=10)
# rlatents = dreduce(latents, levecs)
# dmat = squareform(pdist(rlatents))

n_components = 10
print(
    "constructing PCKDE over small sample images (%d primary components)" % n_components
)
pckde = PCKDE(latents, n_components)

regular_predictions = trainer.predict(regular_images, disable_tqdm=False)
regular_latents = np.array(regular_predictions["latent"])
print("latency data of regular images complete")
reject_predictions = trainer.predict(reject_images, disable_tqdm=False)
reject_latents = np.array(reject_predictions["latent"])
print("latency data of reject images complete")
regular_probs = pckde(regular_latents)
reject_probs = pckde(reject_latents)
print("mean regular prob: %f" % regular_probs.mean())
print("mean reject prob: %f" % reject_probs.mean())


# x = latents.copy()  # [image_index][atom_index*network_index][latent_dimension]
# mean = x.mean(axis=0)
# std = x.std(axis=0)

# image = images[0]
# numbers = image.arrays['numbers']

# n = len(numbers)
# reordered_data = []
# for image_index in range(x.shape[0]):
# reordered_data.append([[], [], []])
# for aid_nid in range(x.shape[1]):
# for i in range(int(x.shape[1]/n)):
# reordered_data[-1][i] = latents[image_index, i*n:(i+1)*n, :]
# latent_dataset = np.moveaxis(np.array(reordered_data), [2, 0, 1, 3], [0, 1, 2, 3])

# ld = latent_dataset.copy()
# ld_elements = {}
# for i, atom_data in enumerate(ld):
# if numbers[i] not in ld_elements:
# ld_elements[numbers[i]] = []
# ld_elements[numbers[i]].append(atom_data)

# for num in ld_elements.keys():
# ld_elements[num] = np.moveaxis(np.array(ld_elements[num]), 2, 0).reshape(3, -1)


# latent_dataset[188][200][3][10]
# latent_dataset[atom_index][image_index][network_index][latent_dimension]

# TODO split latent_data[:] into Cu, Zn, O data --> compare std dev of latent dimensions 3x3 heatmap --> net/element predictions

# Cu latent variance|
# Zn latent variance|
# O  latent variance|______________________
# Cu Net  Zn Net  O Net
