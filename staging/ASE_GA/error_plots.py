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
from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel, RBF, DotProduct
from sklearn.model_selection import train_test_split

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


def condition(training_images):
    config = new_config(EarlyStopping(patience=35, threshold=.10))
    torch.set_num_threads(1)
    trainer = AtomsTrainer(config)
    st = time.time()
    trainer.train(training_images)
    ft = time.time()
    print('trainer conditioned in %0.3f seconds with a dataset of %d images' % (ft - st, len(training_images)))
    return trainer

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


dataset_size = 200
print('training starting')
dataset = sample_images(dataset_size, False, 123)
# trainer = condition(dataset)
config = new_config(EarlyStopping(patience=35, threshold=0.10))
config["dataset"]["raw_data"] = dataset[:1]
trainer = AtomsTrainer(config)
trainer.load_pretrained(
    # "C:\\Users\\ericm\\Documents\\Research\\Code\\Amptorch\\staging\\ASE_GA\\checkpoints\\2021-02-28-00-19-59-test"
    "C:\\Users\\ericm\\Documents\\Research\\Code\\Amptorch\\staging\\ASE_GA\\checkpoints\\2021-03-04-18-46-00-test"
)
trainer.pretrained = False
print("trainer loaded")


predictions = trainer.predict(images, disable_tqdm=False)

true_energies = np.array([image.get_potential_energy() for image in images])  # [:200]])
pred_energies = np.array(predictions["energy"])
abs_errs = np.abs(true_energies - pred_energies)
sqr_errs = np.power(true_energies - pred_energies, 2.)
print("energy MAE:", abs_errs.mean())
print("energy MSE:", sqr_errs.mean())

new_images = []
for i in range(20):
    new_images += read('initial_structure_%d.traj' % i, index=':')
new_predictions = trainer.predict(new_images, disable_tqdm=False)
new_pred_energies = np.array(new_predictions["energy"])
new_true_energies = np.array([image.get_potential_energy() for image in new_images])
new_abs_errs = np.abs(new_true_energies - new_pred_energies)
new_sqr_errs = np.power(new_true_energies - new_pred_energies, 2.)
print("test energy MAE:", new_abs_errs.mean())
print("test energy MSE:", new_sqr_errs.mean())

ml_images = read('ml_opt_active_00.traj', index=':')
ml_predictions = trainer.predict(ml_images, disable_tqdm=False)
ml_pred_energies = np.array(ml_predictions["energy"])
ml_true_energies = np.array([image.get_potential_energy() for image in ml_images])
ml_abs_errs = np.abs(ml_true_energies - ml_pred_energies)
ml_sqr_errs = np.power(ml_true_energies - ml_pred_energies, 2.)
print("ml opt energy MAE:", ml_abs_errs.mean())
print("ml opt energy MSE:", ml_sqr_errs.mean())


import matplotlib.pyplot as plt

def calculate_metrics(latent_data, pde):
    dmat = squareform(pdist(latent_data))
    inv_dmat = np.power((dmat + np.diag([np.inf for _ in range(len(dmat))])), -1.)
    
    point_densities = np.power(inv_dmat.sum(axis=0) / (len(inv_dmat) - 1.), -1.)
        
    probs = np.power(pde(latent_data), -1)

    n_points = 10
    n_nearest = np.array([dmat[i].argsort()[1:n_points+1] for i in range(len(dmat))])
    n_nearest_distances = np.array([[dmat[i, j] for j in n_nearest[i]] for i in range(len(dmat))])
    nnn_distance_averages = n_nearest_distances.mean(axis=1)

    inv_nnn_distances = 1./n_nearest_distances
    inv_nnn_distance_averages = inv_nnn_distances.mean(axis=1)
    inv_nnn_distance_averages = np.power(inv_nnn_distance_averages, -1.)
    
    print('metrics calculated')
    return point_densities, probs, nnn_distance_averages, inv_nnn_distance_averages


latents = np.array(predictions['latent'])
kde = PCKDE(latents)
new_latents = np.array(new_predictions['latent'])
ml_latents = np.array(ml_predictions['latent'])

print('main')
point_densities, probs, nnn_distance_averages, inv_nnn_distance_averages = calculate_metrics(latents, kde)

print('new')
new_point_densities, new_probs, new_nnn_distance_averages, new_inv_nnn_distance_averages = calculate_metrics(new_latents, kde)

print('ml opt')
ml_point_densities, ml_probs, ml_nnn_distance_averages, ml_inv_nnn_distance_averages = calculate_metrics(ml_latents, kde)


def condition_error_gpr(x, y, kernel, label, seed, extension=0.):
    x = x.reshape(-1, 1)
    gpr = GaussianProcessRegressor(kernel=kernel)
    x_train, x_test, y_train, y_test = train_test_split(x, Y, test_size=.75, random_state=seed)
    gpr.fit(x_train, y_train)
    print(label, gpr.kernel_, gpr.log_marginal_likelihood_value_)
    y_pred = gpr.predict(x_test)
    if extension > 0.:
        x_min, x_max = x.min(), x.max()
        x_range = x_max - x_min
        x_lb = x_min - x_range * .1
        x_ub = x_max + x_range * .1
        x_points = np.concatenate([np.linspace(x_lb, x_min, 50), np.linspace(x_max, x_ub, 50)]).reshape(-1, 1)
        y_pred2 = gpr.predict(x_points)
        return x_train, x_test, y_train, y_test, gpr, y_pred, x_points, y_pred2
    else:
        return x_train, x_test, y_train, y_test, gpr, y_pred, None, None


def plot_error_gpr(ax, label, x_train, x_test, y_train, y_test, y_pred, x_points, y_pred2):
    if (x_train is not None) and (y_train is not None):
        ax.scatter(x_train, y_train, alpha=.1, c='b', label=label+' train')
    if (x_test is not None) and (y_pred is not None):
        ax.scatter(x_test, y_pred, alpha=.1, c='y', label=label+' pred')
    if (x_test is not None) and (y_test is not None):
        ax.scatter(x_test, y_test, alpha=.1, s=10, c='r', label=label+' true')
    if (x_points is not None) and (y_pred2 is not None):
        ax.scatter(x_points, y_pred2, alpha=.25, s=16, c='g', label=label+' bounds')
    ax.legend()
    print("pl't")


fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)

lin_kernel = WhiteKernel() + ConstantKernel() * DotProduct()
qdr_kernel = WhiteKernel() + ConstantKernel() * DotProduct() + ConstantKernel() * DotProduct() ** 2
rbf_kernel = WhiteKernel() + ConstantKernel() * RBF()

data = [
    [point_densities, new_point_densities, ml_point_densities, 'LD'],
    [probs, new_probs, new_point_densities, 'LP'],
    [nnn_distance_averages, new_nnn_distance_averages, new_inv_nnn_distance_averages, 'KND'],
    [inv_nnn_distance_averages, new_inv_nnn_distance_averages, ml_inv_nnn_distance_averages, 'KID'],
]

seed = 123
Y = abs_errs.reshape(-1, 1)
new_Y = new_abs_errs.reshape(-1, 1)
ml_Y = ml_abs_errs.reshape(-1, 1)

figs = []
# for i in range(4):
fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)
axs = [ax0, ax1, ax2, ax3]

for i, (main, new, ml_opt, label) in enumerate(data):
    lin_x_train, lin_x_test, lin_y_train, lin_y_test, lin_gpr, lin_y_pred, _, __ = condition_error_gpr(main, Y, lin_kernel, label, seed)
    # qdr_x_train, qdr_x_test, qdr_y_train, qdr_y_test, qdr_gpr, qdr_y_pred, _, __ = condition_error_gpr(main, Y, qdr_kernel, label, seed)
    # rbf_x_train, rbf_x_test, rbf_y_train, rbf_y_test, rbf_gpr, rbf_y_pred, _, __ = condition_error_gpr(main, Y, rbf_kernel, label, seed)
    # plot_error_gpr(axs[i], label, lin_x_train, lin_x_test, lin_y_train, lin_y_test, lin_y_pred, None, None)
    plot_error_gpr(axs[i], label, lin_x_train, None, lin_y_train, None, None, None, None)
    # plot_error_gpr(axs[i], label, None, qdr_x_test, None, None, qdr_y_pred, None, None)
    # plot_error_gpr(axs[i], label, None, rbf_x_test, None, None, rbf_y_pred, None, None)

fig.show()
# fname = input()
# if fname:
#     plt.savefig(fname)



