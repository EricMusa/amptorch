"""data_preprocess.py: Defines the PyTorch required Dataset class. Takes in raw
training data, computes/scales image fingerprints and potential energies.
Functions are included to factorize the data and organize it accordingly in
'collate_amp' as needed to be fed into the PyTorch required DataLoader class"""

import sys
import copy
import os
import numpy as np
import torch
from torch.utils.data import Dataset, SubsetRandomSampler
from amp.utilities import hash_images
from amp.model import calculate_fingerprints_range
from amp.descriptor.gaussian import Gaussian
import ase

__author__ = "Muhammed Shuaibi"
__email__ = "mshuaibi@andrew.cmu.edu"


class AtomsDataset(Dataset):
    """
    PyTorch inherited abstract class that specifies how the training data is to be preprocessed and
    passed along to the DataLoader.

    Parameters:
    -----------
    images: list, file, database
        Training data to be utilized for the regression model.

    descriptor: object
        Scheme to be utilized for computing fingerprints

    forcetraining: float
        Flag to specify whether force training is to be performed - dataset
        will then compute the necessary information - fingerprint derivatives,
        etc.

    """

    def __init__(self, images, descriptor, forcetraining):
        self.images = images
        self.descriptor = descriptor
        self.atom_images = self.images
        self.forcetraining = forcetraining

        if isinstance(images, str):
            extension = os.path.splitext(images)[1]
            if extension != (".traj" or ".db"):
                self.atom_images = ase.io.read(images, ":")
        self.hashed_images = hash_images(self.atom_images)
        self.descriptor.calculate_fingerprints(
            self.hashed_images, calculate_derivatives=forcetraining
        )
        self.fprange = calculate_fingerprints_range(self.descriptor, self.hashed_images)

    def __len__(self):
        return len(self.hashed_images)

    def __getitem__(self, index):
        hash_name = list(self.hashed_images.keys())[index]
        image_fingerprint = self.descriptor.fingerprints[hash_name]
        fprange = self.fprange
        # fingerprint scaling to [-1,1]
        for i, (atom, afp) in enumerate(image_fingerprint):
            _afp = copy.copy(afp)
            fprange_atom = fprange[atom]
            for _ in range(np.shape(_afp)[0]):
                if (fprange_atom[_][1] - fprange_atom[_][0]) > (10.0 ** (-8.0)):
                    _afp[_] = -1 + 2.0 * (
                        (_afp[_] - fprange_atom[_][0])
                        / (fprange_atom[_][1] - fprange_atom[_][0])
                    )
            image_fingerprint[i] = (atom, _afp)
        try:
            image_potential_energy = self.hashed_images[hash_name].get_potential_energy(
                apply_constraint=False
            )
            if self.forcetraining:
                image_forces = self.hashed_images[hash_name].get_forces(
                    apply_constraint=False
                )
                image_primes = self.descriptor.fingerprintprimes[hash_name]
                # fingerprint derivative scaling to [0,1]
                _image_primes = copy.copy(image_primes)
                for _, key in enumerate(list(image_primes.keys())):
                    base_atom = key[3]
                    fprange_atom = fprange[base_atom]
                    fprime = image_primes[key]
                    for i in range(len(fprime)):
                        if (fprange_atom[i][1] - fprange_atom[i][0]) > (10.0 ** (-8.0)):
                            fprime[i] = 2.0 * (
                                fprime[i] / (fprange_atom[i][1] - fprange_atom[i][0])
                            )
                    _image_primes[key] = fprime

                image_prime_values = list(_image_primes.values())
                image_prime_keys = list(_image_primes.keys())
                fp_length = len(image_fingerprint[0][1])
                num_atoms = len(image_fingerprint)
                fingerprintprimes = torch.zeros(fp_length * num_atoms, 3 * num_atoms)
                for idx, fp_key in enumerate(image_prime_keys):
                    image_prime = torch.tensor(image_prime_values[idx])
                    base_atom = fp_key[2]
                    wrt_atom = fp_key[0]
                    coord = fp_key[4]
                    fingerprintprimes[
                        base_atom * fp_length : base_atom * fp_length + fp_length,
                        wrt_atom * 3 + coord,
                    ] = image_prime

        except NotImplementedError:
            print("Atoms object has no claculator set!")
        return (
            (
                image_fingerprint,
                image_potential_energy,
                _image_primes,
                image_forces,
                fingerprintprimes,
                self.forcetraining,
            )
            if self.forcetraining
            else (image_fingerprint, image_potential_energy, self.forcetraining)
        )

    def scalings(self):
        """Computes the scaling factors used in the training dataset to
        standardize the target values. Computed scaling factors will be
        required to scale test sets in the same manner."""
        energy_dataset = []
        for image in self.hashed_images.keys():
            energy_dataset.append(
                self.hashed_images[image].get_potential_energy(apply_constraint=False)
            )
        energy_dataset = torch.tensor(energy_dataset)
        scaling_mean = torch.mean(energy_dataset)
        scaling_sd = torch.std(energy_dataset, dim=0)
        return [scaling_sd, scaling_mean]

    def unique(self):
        """Returns the unique elements contained in the training dataset"""
        return list(self.fprange.keys())

    def fp_length(self):
        """Computes the fingerprint length of the training images"""
        return len(list(self.fprange.values())[0])

    def create_splits(self, training_data, val_frac):
        """Constructs a training and validation sampler to be utilized for
        constructing training and validation sets."""
        dataset_size = len(training_data)
        indices = np.random.permutation(dataset_size)
        split = int(val_frac * dataset_size)
        train_idx, val_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        samplers = {"train": train_sampler, "val": val_sampler}

        return samplers

def factorize_data(training_data):
    """
    Factorizes the dataset into separate lists.

    1. unique_atoms = Identifies the unique elements in the dataset

    2. fingerprint_dataset = Extracts the fingerprints for each hashed data
    sample in the dataset

    3. energy_dataset = Extracts the ab initio potential energy for each hashed data
    sample in the dataset

    4. num_of atoms = Number of atoms per image

    5. sparse_fprimes = fingerprint derivatives for each atom across all
    images. Dimensions of said tensor is QxPx3:
        Q - Atoms in batch
        P - Length of fingerprints
        3 - x,y,z components of the fingerprint derivatives. It should be noted
        that for each image containing N atoms, there are 3N directional
        components for each atom. For each atom, N contributions are summed for
        each directional component to arrive at a dimension of 3 for each atom.

    6. image_forces = Extracts the ab initio forces for each hashed data sample in the
    dataset.

    """
    forcetraining = training_data[0][-1]
    unique_atoms = []
    fingerprint_dataset = []
    energy_dataset = []
    num_of_atoms = []
    fp_length = len(training_data[0][0][0][1])
    for image in training_data:
        num_atom = float(len(image[0]))
        num_of_atoms.append(num_atom)
    atoms_in_batch = int(sum(num_of_atoms))
    image_forces = None
    sparse_fprimes = None
    # Construct a sparse matrix with dimensions PQx3Q, if forcetraining is on.
    if forcetraining:
        image_forces = []
        fprimes = torch.zeros(fp_length * atoms_in_batch, 3 * atoms_in_batch)
        dim1_start = 0
        dim2_start = 0
    for idx, image in enumerate(training_data):
        image_fingerprint = image[0]
        fingerprint_dataset.append(image_fingerprint)
        for atom in image_fingerprint:
            element = atom[0]
            if element not in unique_atoms:
                unique_atoms.append(element)
        image_potential_energy = image[1]
        energy_dataset.append(image_potential_energy)
        if forcetraining:
            fprime = image[4]
            dim1 = fprime.shape[0]
            dim2 = fprime.shape[1]
            fprimes[
                dim1_start : dim1 + dim1_start, dim2_start : dim2 + dim2_start
            ] = fprime
            dim1_start += dim1
            dim2_start += dim2
            image_forces.append(torch.from_numpy(image[3]))
    if forcetraining:
        sparse_fprimes = fprimes.to_sparse()
        image_forces = torch.cat(image_forces).float()

    return (
            unique_atoms,
            fingerprint_dataset,
            energy_dataset,
            num_of_atoms,
            sparse_fprimes,
            image_forces
        )

def collate_amp(training_data):
    """
    Reshuffling scheme that reads in raw data and organizes it into element
    specific datasets to be fed into element specific neural networks.
    """

    unique_atoms, fingerprint_dataset, energy_dataset, num_of_atoms, fp_primes, image_forces = factorize_data(
        training_data
    )
    element_specific_fingerprints = {}
    model_input_data = []
    for element in unique_atoms:
        element_specific_fingerprints[element] = [[], []]
    for fp_index, sample_fingerprints in enumerate(fingerprint_dataset):
        for fingerprint in sample_fingerprints:
            atom_element = fingerprint[0]
            atom_fingerprint = fingerprint[1]
            element_specific_fingerprints[atom_element][0].append(
                torch.tensor(atom_fingerprint)
            )
            element_specific_fingerprints[atom_element][1].append(fp_index)
    for element in unique_atoms:
        element_specific_fingerprints[element][0] = torch.stack(
            element_specific_fingerprints[element][0]
        )
    model_input_data.append(element_specific_fingerprints)
    model_input_data.append(torch.tensor(energy_dataset))
    model_input_data.append(torch.tensor(num_of_atoms))
    model_input_data.append(fp_primes)
    model_input_data.append(image_forces)
    return model_input_data

class TestDataset(Dataset):
    """
    PyTorch inherited abstract class that specifies how testing data is to be preprocessed and
    passed along to the DataLoader.

    Parameters:
    -----------
    images: list, file, database
        Training data to be utilized for the regression model.

    descriptor: object
        Scheme to be utilized for computing fingerprints

    fprange: Dict
        Fingerprint ranges of the training dataset to be used to scale the test
        dataset fingerprints in the same manner.
    """

    def __init__(self, images, descriptor, fprange):
        self.images = [images]
        self.descriptor = descriptor
        self.atom_images = self.images
        if isinstance(images, str):
            extension = os.path.splitext(images)[1]
            if extension != (".traj" or ".db"):
                self.atom_images = ase.io.read(images, ":")
        self.hashed_images = hash_images(self.atom_images)
        self.descriptor.calculate_fingerprints(
            self.hashed_images, calculate_derivatives=True
        )
        self.fprange = fprange

    def __len__(self):
        return len(self.hashed_images)

    def __getitem__(self, index):
        hash_name = list(self.hashed_images.keys())[index]
        image_fingerprint = self.descriptor.fingerprints[hash_name]
        fprange = self.fprange
        # fingerprint scaling to a range of [-1,1].
        for i, (atom, afp) in enumerate(image_fingerprint):
            _afp = copy.copy(afp)
            fprange_atom = fprange[atom]
            for _ in range(np.shape(_afp)[0]):
                if (fprange_atom[_][1] - fprange_atom[_][0]) > (10.0 ** (-8.0)):
                    _afp[_] = -1 + 2.0 * (
                        (_afp[_] - fprange_atom[_][0])
                        / (fprange_atom[_][1] - fprange_atom[_][0])
                    )
            image_fingerprint[i] = (atom, _afp)
        image_primes = self.descriptor.fingerprintprimes[hash_name]
        # fingerprint derivative scaling to a range of [0,1].
        _image_primes = copy.copy(image_primes)
        for _, key in enumerate(list(image_primes.keys())):
            base_atom = key[3]
            fprange_atom = fprange[base_atom]
            fprime = image_primes[key]
            for i in range(len(fprime)):
                if (fprange_atom[i][1] - fprange_atom[i][0]) > (10.0 ** (-8.0)):
                    fprime[i] = 2.0 * (
                        fprime[i] / (fprange_atom[i][1] - fprange_atom[i][0])
                    )
            _image_primes[key] = fprime

        image_prime_values = list(_image_primes.values())
        image_prime_keys = list(_image_primes.keys())
        fp_length = len(image_fingerprint[0][1])
        num_atoms = len(image_fingerprint)
        fingerprintprimes = torch.zeros(fp_length * num_atoms, 3 * num_atoms)
        for idx, fp_key in enumerate(image_prime_keys):
            image_prime = torch.tensor(image_prime_values[idx])
            base_atom = fp_key[2]
            wrt_atom = fp_key[0]
            coord = fp_key[4]
            fingerprintprimes[
                base_atom * fp_length : base_atom * fp_length + fp_length,
                wrt_atom * 3 + coord,
            ] = image_prime

        return (image_fingerprint, fingerprintprimes, num_atoms)

    def unique(self):
        self.unique_atoms = list(self.fprange.keys())
        return self.unique_atoms

    def fp_length(self):
        self.fp_length = len(list(self.fprange.values())[0])
        return len(list(self.fprange.values())[0])

    def collate_test(self, training_data):
        """
        Reshuffling scheme that reads in raw data and organizes it into element
        specific datasets to be fed into the element specific Neural Nets.
        """
        fingerprint_dataset = []
        num_of_atoms = []
        fp_length = self.fp_length
        for image in training_data:
            num_of_atoms.append(image[2])
        atoms_in_batch = int(sum(num_of_atoms))
        # Construct a sparse matrix with dimensions PQx3Q
        fprimes = torch.zeros(fp_length * atoms_in_batch, 3 * atoms_in_batch)
        dim1_start = 0
        dim2_start = 0
        for idx, image in enumerate(training_data):
            fprime = image[1]
            dim1 = fprime.shape[0]
            dim2 = fprime.shape[1]
            fprimes[
                dim1_start : dim1 + dim1_start, dim2_start : dim2 + dim2_start
            ] = fprime
            dim1_start += dim1
            dim2_start += dim2
            image_fingerprint = image[0]
            fingerprint_dataset.append(image_fingerprint)
        sparse_fprimes = fprimes.to_sparse()

        element_specific_fingerprints = {}
        model_input_data = []
        for element in self.unique_atoms:
            element_specific_fingerprints[element] = [[], []]
        for fp_index, sample_fingerprints in enumerate(fingerprint_dataset):
            for fingerprint in sample_fingerprints:
                atom_element = fingerprint[0]
                atom_fingerprint = fingerprint[1]
                element_specific_fingerprints[atom_element][0].append(
                    torch.tensor(atom_fingerprint)
                )
                element_specific_fingerprints[atom_element][1].append(fp_index)
        for element in self.unique_atoms:
            element_specific_fingerprints[element][0] = torch.stack(
                element_specific_fingerprints[element][0]
            )
        model_input_data.append(element_specific_fingerprints)
        model_input_data.append(num_of_atoms)
        model_input_data.append(sparse_fprimes)

        return model_input_data
