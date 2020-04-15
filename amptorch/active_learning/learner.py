import sys
import os
import random
import numpy as np

import torch

from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator as sp

import skorch
from skorch import NeuralNetRegressor
from skorch.dataset import CVSplit
from skorch.callbacks import Checkpoint, EpochScoring

from amptorch.gaussian import SNN_Gaussian
from amptorch.skorch_model import AMP
from amptorch.skorch_model.utils import target_extractor, energy_score, forces_score, train_end_load_best_loss
from amptorch.data_preprocess import AtomsDataset, collate_amp
from amptorch.model import FullNN, CustomMSELoss
from amptorch.delta_models.morse import morse_potential
from amptorch.active_learning.trainer import bootstrap_ensemble, train_calcs
from amptorch.active_learning.query_methods import termination_criteria


__author__ = "Muhammed Shuaibi"
__email__ = "mshuaibi@andrew.cmu.edu"


class AtomisticActiveLearner(object):
    """Active Learner

    Parameters
    ----------
     parent_calc : object
         ASE parent calculator to be called for active learning queries.

     images: list
         Starting set of training images available.

     filename : str
         Label to save model and generated trajectories.

     file_dir: str
         Directory to store results.
     """

    implemented_properties = ["energy", "forces"]

    def __init__(self,
                 training_data,
                 training_params,
                 parent_calc,
                 ensemble=False):
        self.training_data = training_data
        self.training_params = training_params
        self.parent_calc = parent_calc
        self.ensemble = ensemble
        self.parent_calls = 0

        if ensemble:
            assert(isinstance(ensemble, int) and ensemble > 1), 'Invalid ensemble!'
            self.training_data, self.parent_dataset = bootstrap_ensemble(
                    training_data, n_ensembles=ensemble)
        else:
            self.parent_dataset = self.training_data

    def learn(
        self,
        generating_function,
        query_strategy,
    ):
        al_convergence = self.training_params["al_convergence"]
        samples_to_retrain = self.training_params["samples_to_retrain"]
        filename = self.training_params["filename"]
        file_dir = self.training_params["file_dir"]

        terminate = False
        iteration = 0

        while not terminate:
            fn_label = f'{file_dir}{filename}_iter_{iteration}'
            # active learning random scheme
            if iteration > 0:
                queried_images = query_strategy(
                    self.parent_dataset,
                    sample_candidates,
                    samples_to_retrain,
                    parent_calc=self.parent_calc,
                )
                self.parent_dataset, self.training_data = self.add_data(queried_images)
                self.parent_calls += len(queried_images)

            # train ml calculator
            trained_calc = train_calcs(
                training_data=self.training_data,
                training_params=self.training_params,
                ensemble=self.ensemble,
                ncores=3
            )
            # run generating function using trained ml calculator
            generating_function.run(calc=trained_calc, filename=fn_label)
            # collect resulting trajectory files
            sample_candidates = generating_function.get_trajectory(
                filename=fn_label, start_count=0, end_count=-1, interval=1
            )
            iteration += 1
            # criteria to stop active learning
            #TODO Find a better way to structure this.
            method = al_convergence["method"]
            if method == 'iter':
                 termination_args = {"current_i": iteration, "total_i":
                         al_convergence["num_iterations"]}
            elif method == 'spot':
                termination_args = {"images": sample_candidates, "num2verify":
                        al_convergence["num2verify"]}

            terminate = termination_criteria(method=method,
                    termination_args=termination_args)

    def add_data(self, queried_images):
        if self.ensemble:
            for query in queried_images:
                self.training_data, self.parent_dataset =bootstrap_ensemble(
                        self.parent_dataset, self.training_data, query,
                        n_ensembles=self.ensemble)
        else:
            self.training_data += queried_images
        return self.parent_dataset, self.training_data
