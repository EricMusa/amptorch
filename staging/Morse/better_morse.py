import numpy as np
import json
import os
from ase.calculators.calculator import Calculator


class ParametricMorsePotential(Calculator):
    """Morse potential.

    Default values chosen to be similar as Lennard-Jones.
    """

    implemented_properties = ['energy', 'forces']
    nolabel = True
    mandatory_parameters = ['element_A', 'element_B', 'epsilon', 'r0', 'rho0']

    def __init__(self, morse_parameters, **kwargs):
        """
        Parameters
        ----------
        epsilon: float
          Absolute minimum depth, default 1.0
        r0: float
          Minimum distance, default 1.0
        rho0: float
          Exponential prefactor. The force constant in the potential minimum
          is k = 2 * epsilon * (rho0 / r0)**2, default 6.0
        """
        Calculator.__init__(self, **kwargs)
        self.morse_parameters = self.process_parameters(morse_parameters)

    def process_parameters(self, parameters):
        """expect the json format already stored under /morse_params"""
        assert os.path.isfile(parameters)
        with open(parameters, 'r') as f:
            parameters = json.load(f)
        processed_data = {}
        for pair in parameters:
            elem_key = self.get_elem_key(pair[0], pair[1])
            De, re, a = pair[2:]
            processed_data[elem_key] = [De, a*re, re, 2*De*a]  # [epsilon, rho0, r0, preF]
        return processed_data

    @staticmethod
    def get_elem_key(elem_a, elem_b):
        return ''.join(sorted([elem_a, elem_b]))

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=['positions', 'numbers', 'cell',
                                  'pbc', 'charges', 'magmoms']):
        Calculator.calculate(self, atoms, properties, system_changes)
        energy = 0.0
        forces = np.zeros((len(self.atoms), 3))
        for i1, a1 in enumerate(self.atoms):
            for i2, a2 in enumerate(self.atoms[:i1]):
                elem_key = self.get_elem_key(a1.symbol, a2.symbol)
                epsilon, rho0, r0, preF = self.morse_parameters[elem_key]
                # preF = 2 * epsilon * rho0 / r0
                diff = a2.position - a1.position
                r = np.sqrt(np.dot(diff, diff))
                expf = np.exp(rho0 * (1.0 - r / r0))
                energy += epsilon * expf * (expf - 2)
                F = preF * expf * (expf - 1) * diff / r
                forces[i1] -= F
                forces[i2] += F
        self.results['energy'] = energy
        self.results['forces'] = forces