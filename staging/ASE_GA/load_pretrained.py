import numpy as np
import os
import torch
from amptorch.ase_utils import AMPtorch
from amptorch.descriptor.Gaussian import GaussianDescriptorSet
from amptorch.trainer import AtomsTrainer
from ase.io import read
from gea import *

images = read('mldb.db', index=':')  # 0:10')
print('%d images loaded' % len(images))

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
        "epochs": 5000,
        "loss": "mse",
        "metric": "mae",
        "gpus": 0,
    },
    "dataset": {
        "raw_data": images[:10],
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
trainer.load_pretrained('C:\\Users\\ericm\\Documents\\Research\\Code\\Amptorch\\staging\\ASE_GA\\checkpoints\\2021-02-16-04-24-09-test')
# trainer.train()
print('pretrained model loaded')
predictions = trainer.predict(images[:200], disable_tqdm=False)

# true_energies = np.array([image.get_potential_energy() for image in images])  # [:200]])
# pred_energies = np.array(predictions["energy"])

# print("Energy MSE:", np.mean((true_energies - pred_energies) ** 2))
# print("Energy MAE:", np.mean(np.abs(true_energies - pred_energies)))

# images[0].set_calculator(AMPtorch(trainer))
# images[0].get_potential_energy()

from gea import *
latents = np.array(predictions['latent'])
n_components = 10
pckde = PCKDE(latents, n_components)
probs = pckde(latents)
print(probs.min(), probs.max(), probs.mean(), probs.std())
std_probs = (probs / probs.std())
print(std_probs.min(), std_probs.max(), std_probs.mean(), std_probs.std())
lats = latents[180:184]
results_pcs = pckde.density(lats,)
results_npc = pckde.density(lats, use_pcs=False)
print(4, results_pcs[0])
print(4, results_pcs[1])
# print(4, results_pcs[2])
# print(4, results_pcs[3])

lats = latents[180:185]
results_pcs = pckde.density(lats,)
results_npc = pckde.density(lats, use_pcs=False)
print(5, results_pcs[0])
print(5, results_pcs[1])
# print(5, results_pcs[2])
# print(5, results_pcs[3])