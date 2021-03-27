from ase.md import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize.sciopt import SciPyFminCG, SciPyFminBFGS
from scipy.spatial.distance import squareform, pdist
from staging.ASE_GA.gea import PCKDE, inverse_distance_density, n_nearest_neighbors
import numpy as np
import os
import time
import torch
from amptorch.ase_utils import AMPtorch
from amptorch.descriptor.Gaussian import GaussianDescriptorSet
from amptorch.trainer import AtomsTrainer
from skorch.callbacks import EarlyStopping
from ase.io import read
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    WhiteKernel,
    RBF,
    DotProduct,
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from ase.ga.data import PrepareDB
from ase.ga.startgenerator import StartGenerator
from ase.ga.utilities import closest_distances_generator
from ase.ga.utilities import get_all_atom_types
from ase.constraints import FixAtoms
from ase.build import fcc100
from ase.data import atomic_numbers
from ase.formula import Formula
from ase.visualize import view
from ase import Atoms, units
from ase.calculators.emt import EMT
from better_morse import ParametricMorsePotential


def get_slab():
    slab = fcc100("Au", size=(8, 8, 4), vacuum=12.0, orthogonal=True)
    slab.pbc[2] = True
    slab.set_constraint(
        FixAtoms(indices=[i for i, atom in enumerate(slab) if atom.position[2] < 13.0])
    )
    print("slab created:", slab.get_chemical_formula())
    pos = slab.get_positions()
    cell = slab.get_cell()

    a, b, c = cell[0, 0], cell[1, 1], cell[2, 2]
    p0 = np.array([a * 0.25, b * 0.25, max(pos[:, 2]) + 2.0])
    v1 = np.array([a, 0, 0]) * 0.5
    v2 = np.array([0, b, 0]) * 0.5
    v3 = np.array([0, 0, c]) * 0.25

    cluster_composition = "Pt8Ag8"
    cluster_formula = Formula(cluster_composition)
    print("cluster composition:", cluster_formula)
    atom_numbers = []
    for element, count in cluster_formula.count().items():
        atom_numbers += [atomic_numbers[element]] * count

    unique_atom_types = get_all_atom_types(slab, atom_numbers)
    blmin = closest_distances_generator(
        atom_numbers=unique_atom_types, ratio_of_covalent_radii=0.7
    )

    slab_cluster_sg = StartGenerator(
        slab, atom_numbers, blmin, box_to_place_in=[p0, [v1, v2, v3]]
    )

    return slab, slab_cluster_sg


def optimize(atoms, opt, trajectory, fmax=0.05, steps=200, **kwargs):
    if not os.path.isfile(trajectory):
        atoms.set_calculator(EMT())
        print(
            "optimizing geometry of",
            atoms.get_chemical_formula(),
            "to fmax = %f eV/A0" % fmax,
        )
        opt = opt(atoms, trajectory=trajectory, logfile="-", **kwargs)
        opt.run(fmax=fmax, steps=steps)
    return read(trajectory, index=":")


def dynamics(atoms, dyn, trajectory, temperature=300, steps=200, **kwargs):
    if not os.path.isfile(trajectory):
        atoms.set_calculator(EMT())
        print(
            "running dynamics of",
            atoms.get_chemical_formula(),
            "at %d K and a timestep of 5 fs" % temperature,
        )
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
        dyn = dyn(atoms, 5 * units.fs, trajectory=trajectory, logfile="-", **kwargs)
        dyn.run(steps=steps)
    return read(trajectory, index=":")


def compress_slab_cluster(atoms, cluster_index, compression_factor=0.8):
    assert iter(cluster_index) is not None
    if all(isinstance(_, str) for _ in cluster_index):
        cluster_index = [_.index for _ in atoms if _.symbol in cluster_index]
    cluster_index = np.array(cluster_index)

    atoms = atoms.copy()
    slab_index = np.array([_ for _ in range(len(atoms)) if _ not in cluster_index])
    cluster_pos = atoms[cluster_index].positions
    slab_pos = atoms[slab_index].positions
    com = cluster_pos.mean(axis=0)
    rads = cluster_pos - com
    rads *= compression_factor
    cluster_pos = rads + com
    max_z = slab_pos[:, 2].max()
    min_z = cluster_pos[:, 2].min()
    compression_dist_z = (min_z - max_z) * (1.0 - compression_factor)
    cluster_pos -= compression_dist_z
    atoms.positions[cluster_index] = cluster_pos
    atoms.positions[slab_index] = slab_pos
    return atoms


slab, slab_start_generator = get_slab()

atoms = slab_start_generator.get_new_candidate()
pos149 = atoms.positions[149]
center = (atoms.cell.array * 0.5).sum(axis=1)
vec = center - pos149
atoms2 = atoms.copy()
atoms2.translate(vec)
atoms2.wrap()
view([atoms, atoms2])
input(2)

# atoms = slab_start_generator.get_new_candidate()
# calc = ParametricMorsePotential('pt_ag_au')
# atoms.set_calculator(calc)
# dyn = SciPyFminCG(atoms, trajectory='dft_morse.traj', logfile='-')
# dyn.run(fmax=0.05)
# images = read('dft_morse.traj', index=':')
# view(images)
# input()

# slab_cluster_images = dynamics(slab, VelocityVerlet, trajectory='slab_md.traj', steps=5)[::10]

slab_cluster_images = []
fixed_cluster = slab_start_generator.get_new_candidate()
fixed_cluster.set_constraint(FixAtoms(mask=[_.symbol == "Au" for _ in fixed_cluster]))
fixed_traj = optimize(
    fixed_cluster, SciPyFminCG, trajectory="fixed_slab_cluster_opt.traj"
)
cluster = fixed_cluster.copy()
cluster.set_constraint(FixAtoms(mask=[atom.position[2] < 13.0 for atom in slab]))
traj = optimize(cluster, SciPyFminCG, trajectory="slab_cluster_opt.traj")
compressed_cluster = compress_slab_cluster(cluster, ["Pt", "Ag"])
compressed_traj = optimize(
    compressed_cluster, SciPyFminCG, trajectory="compressed_cluster_opt.traj", steps=50
)
dyn = dynamics(cluster, VelocityVerlet, trajectory="slab_cluster_dyn.traj", steps=200)
slab_cluster_images += traj + compressed_traj + dyn[::10]
print(len(slab_cluster_images), "slab cluster images total")

all_training_images = slab_cluster_images
elements = np.unique([atom.symbol for atom in slab_start_generator.get_new_candidate()])


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
            "num_nodes": 6,
            "batchnorm": False,
            "latent": True,  # LATENT NNP
        },
        "optim": {
            "force_coefficient": 0.25,
            "lr": 2e-2,
            "batch_size": 25,
            "epochs": 500,
            "loss": "mse",
            "metric": "mae",
            "gpus": 1,
            "callbacks": list(callbacks),
        },
        "dataset": {
            "raw_data": None,
            "val_split": 0.2,
            "fp_params": gds,  # either a GDS or the `Gs` dict can be passed here
            "save_fps": True,
            # feature scaling to be used - normalize or standardize
            # normalize requires a range to be specified
            "scaling": {"type": "standardize"},
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


def condition(training_images):
    config = new_config(EarlyStopping(patience=50, threshold=0.05))
    torch.set_num_threads(1)
    trainer = AtomsTrainer(config)
    st = time.time()
    trainer.train(training_images)
    ft = time.time()
    print(
        "trainer conditioned in %0.3f seconds with a dataset of %d images"
        % (ft - st, len(training_images))
    )
    return trainer


# config = new_config(EarlyStopping(patience=50, threshold=0.10))
# config["dataset"]["raw_data"] = images[:1]
# trainer = AtomsTrainer(config)
# trainer.load_pretrained(
#     "C:\\Users\\ericm\\Documents\\Research\\Code\\Amptorch\\staging\\ASE_GA\\checkpoints\\2021-03-15-12-25-33-test"  # 2021-03-15-22-47-44-test
# )
# print("trainer loaded")
input("holup")
trainer = condition(all_training_images)

images = all_training_images
predictions = trainer.predict(images, disable_tqdm=False)

true_energies = np.array([image.get_potential_energy() for image in images])  # [:200]])
pred_energies = np.array(predictions["energy"])
abs_errs = np.abs(true_energies - pred_energies)
sqr_errs = np.power(true_energies - pred_energies, 2.0)
print("energy MAE:", abs_errs.mean())
print("energy MSE:", sqr_errs.mean())

n_cycles = 4
for i in range(n_cycles):
    atoms = read("sample_start_01.traj")
    atoms.pbc[2] = True
    atoms.set_calculator(AMPtorch(trainer))
    traj = "dataset_ml_opt_%02d.traj" % i
    # dyn = SciPyFminCG(atoms, trajectory=traj, logfile="-")
    dyn = SciPyFminBFGS(atoms, trajectory=traj, logfile="-")
    dyn.run(fmax=0.05, steps=10)
    trajectory = read(traj, index=":")
    # view(trajectory)
    trajectory = [t.copy() for t in trajectory]
    for t in trajectory:
        t.set_calculator(EMT())
        t.get_potential_energy()
    print("training")
    all_training_images += trajectory
    trainer.train(all_training_images)
    print("training done")
    predictions = trainer.predict(all_training_images, disable_tqdm=False)
    true_energies = np.array(
        [image.get_potential_energy() for image in all_training_images]
    )  # [:200]])
    pred_energies = np.array(predictions["energy"])
    abs_errs = np.abs(true_energies - pred_energies)
    sqr_errs = np.power(true_energies - pred_energies, 2.0)
    print("energy MAE:", abs_errs.mean())
    print("energy MSE:", sqr_errs.mean())

# trainer.load_pretrained(
#     "C:\\Users\\ericm\\Documents\\Research\\Code\\Amptorch\\staging\\ASE_GA\\checkpoints\\2021-03-15-16-09-06-test"
# )
# 2021-03-18-19-29-04-test  # --> Compression + bare slab images, conditioned w/ ES(steps=50, threshold=0.05)
