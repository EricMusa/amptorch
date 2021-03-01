from staging.ASE_GA.gea import inverse_distance_density, n_nearest_neighbors
import numpy as np
import os
import time
import torch
from amptorch.ase_utils import AMPtorch
from amptorch.descriptor.Gaussian import GaussianDescriptorSet
from amptorch.trainer import AtomsTrainer
from skorch.callbacks import EarlyStopping
from ase.io import read
from gea import *

images = read('mldb.db', index='1:3720')  # 0:10')
print('%d images loaded' % len(images))

elements = np.unique([atom.symbol for atom in images[0]])

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

    gds.batch_add_descriptors(*low_res_g2, important_elements=hi_res_elements)
    gds.batch_add_descriptors(*low_res_g5, important_elements=hi_res_elements)

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
            "epochs": 500,
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


def low_condition(training_images):
    config = new_config(EarlyStopping(patience=25, threshold=.10))
    torch.set_num_threads(1)
    trainer = AtomsTrainer(config)
    st = time.time()
    trainer.train(training_images)
    ft = time.time()
    print('trainer low conditioned in %0.3f seconds with a dataset of %d images' % (ft - st, len(training_images)))
    return trainer, ft - st


def med_condition(training_images):
    config = new_config(EarlyStopping(patience=50, threshold=.05))
    torch.set_num_threads(1)
    trainer = AtomsTrainer(config)
    st = time.time()
    trainer.train(training_images)
    ft = time.time()
    print('trainer medium conditioned in %0.3f seconds with a dataset of %d images' % (ft - st, len(training_images)))
    return trainer, ft - st


def high_condition(training_images):
    config = new_config(EarlyStopping(patience=100, threshold=.01))
    torch.set_num_threads(1)
    trainer = AtomsTrainer(config)
    st = time.time()
    trainer.train(training_images)
    ft = time.time()
    print('trainer high conditioned in %0.3f seconds with a dataset of %d images' % (ft - st, len(training_images)))
    return trainer, ft - st


traj_indices = [[]]
for i, atoms in enumerate(images[1:]):
    if atoms.get_potential_energy() > images[i].get_potential_energy():
        traj_indices.append([i])
    else:
        traj_indices[-1].append(i)
traj_indices = np.array([np.array(t) for t in traj_indices], dtype=object)


def sample_images(n_images, traj_sampled=False, seed=None):
    if seed:
        np.random.seed(seed)
    if traj_sampled:
        trajs_to_sample = {}
        for i in np.random.choice([_ for _ in range(len(traj_indices))], n_images, replace=True):
            if i in trajs_to_sample:
                trajs_to_sample[i] = trajs_to_sample[i] + 1
            else:
                trajs_to_sample[i] = 1
        sample = []
        for traj_i, n_samples in trajs_to_sample.items():
            for i in np.random.choice(traj_indices[traj_i], n_samples, replace=False):
                sample.append([images[i], i])
            print(traj_i, n_samples, [s[1] for s in sample])
        return [s[0] for s in sample]
    else:
        sample = []
        for i in np.random.choice(np.arange(len(images)), n_images, replace=False):
            sample.append([images[i], i])
        print([s[1] for s in sample])
        return [s[0] for s in sample]


trainers = []
for dataset_size in [200]:
    for b in [False]:
        label = '%d image dataset, %straj_sampled, low conditioning' % (dataset_size, '' if b else 'not ')
        print(label, 'training starting')
        dataset = sample_images(dataset_size, b, 123)
        trainer, train_time = low_condition(dataset)
        predictions = trainer.predict(images, disable_tqdm=False)

        true_energies = np.array([image.get_potential_energy() for image in images])  # [:200]])
        pred_energies = np.array(predictions["energy"])
        mse = np.mean((true_energies - pred_energies) ** 2)
        mae = np.mean(np.abs(true_energies - pred_energies))
        print(label, "- energy MSE:", mse)
        print(label, "- energy MAE:", mae)
        trainers.append([trainer, train_time, label, mse, mae])


import matplotlib.pyplot as plt

latents = np.array(predictions['latent'])
density, pds, inv_dmat, dmat = inverse_distance_density(latents)

## statistics on Point Latent Densities
mu, sigma = pds.mean(), pds.std()
spds = (pds - mu) / sigma
mses = (true_energies - pred_energies) ** 2
maes = np.abs(true_energies - pred_energies)

scaled_x = np.arange(len(spds), dtype=np.float64)
scaled_x /= len(spds)
scaled_x *= (spds.max() - spds.min())
scaled_x += spds.min()
plt.scatter(spds, maes, alpha=.1)  # mean-abs error vs. standardized point density
# plt.plot(scaled_x, sorted(maes, reverse=True), 'r')
plt.plot(scaled_x, np.zeros(len(spds)), 'k',)
plt.xlabel('Standardized Point Latent Density')
plt.ylabel('Absolute Error (eV)')
plt.title('Point Error Vs. Point Latent Density')
plt.savefig('point_errors_vs_point_densities.png')
plt.show()

kde = PCKDE(latents)
probs = kde(latents)

## statistics on KDE Probs
mu, sigma = probs.mean(), probs.std()
sprobs = (probs - mu) / sigma

scaled_x = np.arange(len(sprobs), dtype=np.float64)
scaled_x /= len(sprobs)
scaled_x *= (sprobs.max() - sprobs.min())
scaled_x += sprobs.min()
plt.scatter(sprobs, maes, alpha=.1)  # mean-abs error vs. standardized point density
# plt.plot(scaled_x, sorted(maes, reverse=True), 'r')
plt.plot(scaled_x, np.zeros(len(sprobs)), 'k',)
plt.xlabel('Standardized Point KDE Probability')
plt.ylabel('Absolute Error (eV)')
plt.title('Point Error Vs. Point KDE Probability')
plt.savefig('point_errors_vs_point_probabilities.png')
plt.show()


## statistics on N-Nearest Neighbors Distance
nnn_distances = n_nearest_neighbors(latents, 10)
nnn_distance_averages = nnn_distances.mean(axis=1)
# nnn_distance_sum = nnn_distances.sum(axis=1)
snnn_distances = nnn_distance_averages

scaled_x = np.arange(len(snnn_distances), dtype=np.float64)
scaled_x /= len(snnn_distances)
scaled_x *= (snnn_distances.max() - snnn_distances.min())
scaled_x += snnn_distances.min()
plt.scatter(snnn_distances, maes, alpha=.1)  # mean-abs error vs. standardized point density
# plt.plot(scaled_x, sorted(maes, reverse=True), 'r')
plt.plot(scaled_x, np.zeros(len(snnn_distances)), 'k',)
plt.xlabel('Average N-Nearest Neighbors Distance (N=10)')
plt.ylabel('Absolute Error (eV)')
plt.title('Point Error Vs. N-Nearest Neighbors ')
plt.savefig('point_errors_vs_n_nearest_neighbors_average_distance.png')
plt.show()

## statistics on N-Nearest Neighbors Inverted Distance
inv_nnn_distances = 1./nnn_distances
inv_nnn_distance_averages = inv_nnn_distances.mean(axis=1)
inv_snnn_distances = inv_nnn_distance_averages

scaled_x = np.arange(len(snnn_distances), dtype=np.float64)
scaled_x /= len(snnn_distances)
scaled_x *= (snnn_distances.max() - snnn_distances.min())
scaled_x += snnn_distances.min()
plt.scatter(snnn_distances, maes, alpha=.1)  # mean-abs error vs. standardized point density
# plt.plot(scaled_x, sorted(maes, reverse=True), 'r')
plt.plot(scaled_x, np.zeros(len(snnn_distances)), 'k',)
plt.xlabel('Average N-Nearest Neighbors Inverse Distance (N=10)')
plt.ylabel('Absolute Error (eV)')
plt.title('Point Error Vs. N-Nearest Neighbors (Inverse Distance)')
plt.savefig('point_errors_vs_inversed_n_nearest_neighbors_average_distance.png')
plt.show()


# latents = np.array(predictions['latent'], dtype=np.float32)

# n_components = 10
# pckde = PCKDE(latents, n_components)
# probs = pckde(latents)

# traj_latents = [[latents[i-1] for i in traj] for traj in traj_indices[:n_trajs]]
# indices = [_-1 for inds in traj_indices[:n_trajs] for _ in inds]
# all_results = []
# ends = [traj[-1] for traj in traj_indices]
# x = []
# x_ends = []
# for i in range(20, len(latents)):
#     lats = latents[:i]
#     if i in ends:
#         print(i, len(lats))
#         x_ends.append(i)
#     all_results.append(pckde.density(lats)[:2])
#     x.append(i)


# fig, (ax0, ax1) = plt.subplots(2)

# ax0.scatter(x, [ar[0] for ar in all_results], c='b', label='density')
# ax0.scatter(x_ends, [all_results[i-20][0] for i in x_ends], c='r', label='new relaxation (dens)')
# ax0.set_xlabel('Number of images in the dataset')
# ax0.set_ylabel('Cumulative Latent Density')
# # plt.legend()
# # plt.savefig('latent_density_plot.png')
# # plt.show()

# ax1.scatter(x, probs[x], c='y', label='probability')
# ax1.scatter(x_ends, [probs[i] for i in x_ends], c='r', label='new relaxation (prob)')
# ax1.set_xlabel('Number of images in the dataset')
# ax1.set_ylabel('Latent Probability')
# fig.legend()
# fig.tight_layout()
# fig.savefig('latent_prob_and_density_plot5.png')
# fig.show()

