from ase import Atoms, Atom
from ase.calculators.vasp import Vasp2
from ase.data import atomic_numbers, atomic_names, atomic_masses, covalent_radii
from ase.io import read, write
import numpy as np
import os
import time
from scipy.optimize import curve_fit
from itertools import combinations
import json

dirnames = [dirname for dirname in os.listdir(".") if os.path.isdir(dirname)]

calcs = {}

for dirname in dirnames:
    try:
        calcs[dirname] = read(os.path.join(dirname, "vasprun.xml"))
        print(dirname, calcs[dirname])
    except:
        pass


def pair_dft(element_a, element_b, dist, relaxation):
    calc_name = "%s%s_pair_%08dpm" % tuple(
        sorted([element_a, element_b]) + [int(100 * dist)]
    )
    return calcs[calc_name]


def single_dft(element):
    calc_name = "%s_single" % element
    return calcs[calc_name]


def morse_potential(r, De, re, a):
    # calculate the morse potential energy
    return De * (np.exp(-2 * a * (r - re)) - 2 * np.exp(-a * (r - re)))


def bond_calcs(element_a, element_b):

    # Get the equilibrium bond distance/energy
    atom_a = single_dft(element_a)
    atom_b = single_dft(element_b)
    pair_ab = pair_dft(element_a, element_b, 3.0, True)

    # Calculate the first two morse parameters from the ground state
    De = -(
        pair_ab.get_potential_energy()
        - atom_a.get_potential_energy()
        - atom_b.get_potential_energy()
    )
    re = pair_ab.get_distance(0, 1)

    # Select points around the equilibrium radius to sample
    distances = np.array([0.7, 0.8, 0.9, 1.1, 1.2, 1.5]) * re

    # Stretch the bond and fit the last parameter
    point_calcs = [pair_dft(element_a, element_b, dist, False) for dist in distances]
    stretched_energies = np.array(
        [
            point.get_potential_energy()
            - atom_a.get_potential_energy()
            - atom_b.get_potential_energy()
            for point in point_calcs
        ]
    )

    if De < 0:
        raise RuntimeError(element_a, element_b, "wtf, %f" % De)
    # De=1e-1
    # stretched_energies=stretched_energies \
    #                     -(relaxed_atoms.get_potential_energy()-2*lone_energy) \
    #                     -De
    popt, pcov = curve_fit(
        lambda x, sig: morse_potential(x, De, re, sig),
        distances,
        stretched_energies,
        p0=(2),
    )
    sig = popt[0]
    print("%s-%s: De: %f, re: %f, a: %f" % (element_a, element_b, De, re, sig))
    return element_a, element_b, De, re, sig


def run_calculations(elements):
    combos = sorted(
        set([tuple(sorted(combo)) for combo in combinations(elements * 2, 2)])
    )
    print(combos)
    all_results = []
    for combo in combos:
        print("beginning", combo)
        st = time.time()
        results = list(bond_calcs(*combo))
        ft = time.time()
        print(combo, "finished in %f seconds" % (ft - st))
        all_results.append(results)

    print(list(calcs.keys()))
    return all_results


morse_data_dir = "morse_params"
if not os.path.isdir(morse_data_dir):
    os.makedirs(morse_data_dir)

cu_zn_o_morse_data = run_calculations(["Cu", "Zn", "O"])
with open(os.path.join(morse_data_dir, "cu_zn_o_morse_data.json"), "w") as f:
    json.dump(cu_zn_o_morse_data, f)

pt_ce_o_morse_data = run_calculations(["Pt", "Ce", "O"])
with open(os.path.join(morse_data_dir, "pt_ce_o_morse_data.json"), "w") as f:
    json.dump(pt_ce_o_morse_data, f)

rh_ti_o_morse_data = run_calculations(["Rh", "Ti", "O"])
with open(os.path.join(morse_data_dir, "rh_ti_o_morse_data.json"), "w") as f:
    json.dump(rh_ti_o_morse_data, f)
