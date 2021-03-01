import numpy as np
import os
import torch
from amptorch.ase_utils import AMPtorch
from amptorch.descriptor.Gaussian import GaussianDescriptorSet
from amptorch.trainer import AtomsTrainer
from skorch.callbacks import EarlyStopping
from ase.io import read
from gea import *

all_images = read('mldb.db', index=':')  # 0:10')
energies = []
indices = []
for i, image in enumerate(all_images):
    e = image.get_potential_energy() 
    if e not in energies:
        # print('new energy', e)
        energies.append(e)
    else:
        print(i, 'in energies already, dupe')
        indices.append(i)
for i in indices[::-1]:
    all_images.pop(i)
    print('popped', i)

print('%d images loaded' % len(all_images))

traj_indices = []
for i, atoms in enumerate(all_images[1:]):
    if i == 0:
        traj_indices.append({1})
        continue
    if atoms.get_potential_energy() > all_images[i].get_potential_energy():
        traj_indices.append({i})
    else:
        traj_indices[-1].add(i)
traj_indices = [sorted(t) for t in traj_indices]
n_trajs = 5
images = [all_images[i] for traj in traj_indices[:n_trajs] for i in traj]

elements = np.unique([atom.symbol for atom in images[0]])
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

print("constructed GDS hash:", gds.descriptor_setup_hash)

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
        "batch_size": 50,
        "epochs": 100,
        "loss": "mse",
        "metric": "mae",
        "gpus": 0,
        # "callbacks": [EarlyStopping(patience=25, threshold=.05)],
    },
    "dataset": {
        "raw_data": images[:25],
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

torch.set_num_threads(1)
trainer = AtomsTrainer(config)
# trainer.load_pretrained('C:\\Users\\ericm\\Documents\\Research\\Code\\Amptorch\\staging\\ASE_GA\\checkpoints\\2021-02-16-04-24-09-test')
trainer.train()
# print('pretrained model loaded')
predictions = trainer.predict(images, disable_tqdm=False)

# true_energies = np.array([image.get_potential_energy() for image in images])  # [:200]])
# pred_energies = np.array(predictions["energy"])

# print("Energy MSE:", np.mean((true_energies - pred_energies) ** 2))
# print("Energy MAE:", np.mean(np.abs(true_energies - pred_energies)))

# images[0].set_calculator(AMPtorch(trainer))
# images[0].get_potential_energy()

latents = np.array(predictions['latent'], dtype=np.float32)
traj_latents = [[latents[i-1] for i in traj] for traj in traj_indices[:n_trajs]]
indices = [_-1 for inds in traj_indices[:n_trajs] for _ in inds]
n_components = 10
pckde = PCKDE(latents, n_components)

all_results = []
ends = [traj[-1] for traj in traj_indices]
x = []
for i in range(20, len(latents)):
    lats = latents[:i+1]
    if i in ends:
        print(i, len(lats))
    all_results.append(pckde.density(lats)[:2])
    x.append(i)


import matplotlib.pyplot as plt

plt.plot(x, [ar[0] for ar in all_results], label='density')
plt.scatter(ends[:n_trajs], [all_results[i-1-20][0] for i in ends[:n_trajs]], c='r', label='new relaxation')
plt.xlabel('Number of images in the dataset')
plt.ylabel('Cumulative Latent Density')
plt.legend()
plt.savefig('latent_density_plot.png')
plt.show()

# trajs = []
# lats = []
# for indices in traj_indices:
#     if len(indices) > 100:
#         trajs.append([images[index] for index in indices])
#         lats.append([latents[index] for index in indices])

# all_results = []
# for i in range(len(trajs)):
#     sample = [_ for traj in trajs[:i+1] for _ in traj]
#     print('sample %d: %d images' % (i, len(sample)))
#     sample_lats = np.array([_ for lat in lats[:i+1] for _ in lat])
#     all_results.append(pckde.density(sample_lats))
