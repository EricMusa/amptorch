from ase import Atoms, db
from staging.ASE_GA.ga_initialize import init_slab_cluster_ga
from staging.ASE_GA.ga_operations import (
    init_ga_connection,
    relax_atoms_emt,
    relax_unrelaxed_candidates,
    generate_relaxed_candidate,
)
from staging.ASE_GA.predictor import Predictor, HighPredictedError
import numpy as np
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


cluster_composition = "Pt6Ag6"
db_file = "gaxml2.db"
sg = init_slab_cluster_ga(cluster_composition, db_file=db_file)

population_size = 20
population, da, comp, pairing, mutations, init_data = init_ga_connection(
    db_file, population_size
)

data = []
n_initial = 20
init_structure_label = "initial_structure2_%d"
init_images = []
for i in range(n_initial):
    label = init_structure_label % i
    if not os.path.isfile(label + ".traj"):
        relax_atoms_emt(
            sg.get_new_candidate(),
            trajectory=label + ".traj",
            logfile="-",
            max_steps=20,
        )
    data = [label, read(label + ".traj", index=":")]
    data.append(len(data[1]))
    init_images.append(data)

images = [_ for data in init_images for _ in data[1]]
elements = np.unique([atom.symbol for atom in images[0]])

trainer, train_time = condition(images, EarlyStopping(patience=35, threshold=0.10))

# config = new_config(EarlyStopping(patience=50, threshold=0.10))
# config["dataset"]["raw_data"] = images[:1]
# trainer = AtomsTrainer(config)
# trainer.load_pretrained(
#     # "C:\\Users\\ericm\\Documents\\Research\\Code\\Amptorch\\staging\\ASE_GA\\checkpoints\\2021-02-28-00-19-59-test"
#     "C:\\Users\\ericm\\Documents\\Research\\Code\\Amptorch\\staging\\ASE_GA\\checkpoints\\2021-02-28-21-53-54-test"
# )
# trainer.pretrained = False
# print("trainer loaded")

# trainer, train_time = condition(images, EarlyStopping(patience=50, threshold=0.10))
# trainer.train()
print("trainer loaded")

n_components = 10
kernel = WhiteKernel() + (ConstantKernel() * DotProduct())
ep = Predictor(
    trainer, images, n_components=n_components, kernel=kernel, disable_tqdm=False
)
print("Error Predictor made")


class CaptureImages:
    def __init__(self, atoms):
        self.atoms = atoms
        self.captured_images = [atoms.copy()]
    def write(self):
        self.captured_images.append(self.atoms.copy())


test_atoms = read("sample_start_01.traj")  # sg.get_new_candidate()

# def ml_optimize_attempt(atoms, trajectory=None, logfile=None):
atoms = test_atoms
traj_path = 'ml_opt_active2_%02d.traj'
logfile = '-'
capturer = CaptureImages(atoms)
count = 0
while True:
    traj_path % count
    try:
        print('running optimization')
        dyn = SciPyFminBFGS(ep.connect_opt_atoms(atoms), trajectory=traj_path, logfile=logfile)
        dyn.attach(ep)
        dyn.attach(capturer)
        dyn.run(fmax=0.05, steps=100)
        break
    except HighPredictedError as e:
        print(e)
        uncertain_atoms = atoms.copy()
        uncertain_atoms.set_calculator(EMT())
        uncertain_atoms.get_potential_energy()
        if not uncertain_atoms.get_potential_energy() in ep.true_energies:
            print('sampling uncertain point, retraining, and re-optimizing')
            print('atom true energy', uncertain_atoms.get_potential_energy())
            images.append(uncertain_atoms)
        print('retraining model')
        trainer.config["dataset"]["raw_data"] = images
        trainer.train(images)
        print('recreating predictor')
        ep = Predictor(trainer, images, n_components=n_components, kernel=kernel, disable_tqdm=False)
        traj_images = read(traj_path, index=":")
        if isinstance(traj_images, list):
            if len(traj_images) > 1:
                atoms = traj_images[-2]
            else:
                atoms = traj_images[0]
        else:
            atoms = traj_images
        assert isinstance(atoms, Atoms)

        