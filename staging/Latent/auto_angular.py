import numpy as np
import os
import torch
from amptorch.ase_utils import AMPtorch
from amptorch.descriptor.Gaussian import GaussianDescriptorSet
from amptorch.trainer import AtomsTrainer
from ase.io import read

relaxations = {}
for fname in os.listdir("cu8_on_zno"):
    if fname.endswith(".traj"):
        print("loading %s" % fname)
        relaxations[fname.replace(".traj", "")] = read(
            os.path.join("cu8_on_zno", fname), index=":"
        )
        print(
            "%s loaded, %d images"
            % (fname, len(relaxations[fname.replace(".traj", "")]))
        )

print("total images:", sum(len(v) for v in relaxations.values()))
all_images = [_ for rel in relaxations.values() for _ in rel]
seed = 123
np.random.seed(seed)
sample_size = 200
print("selecting subset of %d images for training" % sample_size)
images = np.random.choice(np.arange(len(all_images)), size=sample_size)
images = [all_images[i] for i in images]

elements = np.unique([atom.symbol for atom in images[0]])
cutoff = 6.0
cosine_cutoff_params = {"cutoff_func": "cosine"}


gds = GaussianDescriptorSet(elements, cutoff, cosine_cutoff_params)

low_res_g2 = (2, [0.025, 0.025], [0.0, 3.0])
# low_res_g5 = (5, [.001, .001], [2.0, 2.0], [-1., 1.])
low_res_g5 = (5, [0.001], [1.0], [1.0])
low_res_elements = ["O", "Zn"]

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
hi_res_elements = ["Cu"]

gds.batch_add_descriptors(*low_res_g2, important_elements=hi_res_elements)
gds.batch_add_descriptors(*low_res_g5, important_elements=hi_res_elements)

print("constructed GDS hash:", gds.descriptor_setup_hash)

config = {
    "model": {
        "get_forces": True,
        "num_layers": 3,
        "num_nodes": 5,
        "batchnorm": False,
    },
    "optim": {
        "force_coefficient": 0.2,
        "lr": 1e-2,
        "batch_size": 32,
        "epochs": 100,
        "loss": "mse",
        "metric": "mae",
        "gpus": 0,
    },
    "dataset": {
        "raw_data": images,
        "val_split": 0.1,
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
trainer.train()

predictions = trainer.predict(images)

true_energies = np.array([image.get_potential_energy() for image in images])
pred_energies = np.array(predictions["energy"])

print("Energy MSE:", np.mean((true_energies - pred_energies) ** 2))
print("Energy MAE:", np.mean(np.abs(true_energies - pred_energies)))

images[0].set_calculator(AMPtorch(trainer))
images[0].get_potential_energy()
