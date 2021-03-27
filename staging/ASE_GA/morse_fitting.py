from ase import Atoms, Atom
from ase.calculators.vasp import Vasp2
from ase.calculators.emt import EMT
from ase.optimize.sciopt import SciPyFminCG
from ase.data import atomic_numbers, atomic_names, atomic_masses, covalent_radii
from ase.io import read, write
import numpy as np
import os
import time
from scipy.optimize import curve_fit
from itertools import combinations

dirnames = [dirname for dirname in os.listdir(".") if os.path.isdir(dirname)]

calcs = {}

for dirname in dirnames:
    try:
        calcs[dirname] = read(os.path.join(dirname, "vasprun.xml"))
        print(dirname, calcs[dirname])
    except:
        pass


def get_cu_on_zno_settings(relaxation=True):
    return {
        "command": "srun -E vasp_std",
        "algo": "Fast",
        "ediff": 0.0001,
        "encut": 500.0,
        "gga": "PE",
        "potim": 0.10,
        "ibrion": (3 if relaxation else 1),  # relaxation or point calculation
        "icharg": 2.0,
        "isif": 0.0,
        "ismear": 0.0,
        "lasph": True,
        "lcharg": False,
        "lreal": "A",
        "lwave": False,
        "nelm": 200.0,
        "nelmdl": -5.0,
        "nelmin": 2.0,
        "nsim": 8.0,
        "nsw": (800 if relaxation else 1),  # relaxation or point calculation
        "prec": "Accurate",
        "sigma": 1.0,
    }


def get_rh_on_tio2_settings(relaxation=True):
    return {
        "command": "srun -E vasp_std",
        "lcharg": False,
        "lwave": False,
        "prec": "Normal",
        "encut": 300,
        "lmaxmix": 4,
        "icharg": 1,
        "idipol": 3,
        "ispin": 2,
        "lorbit": 11,
        "ivdw": 11,
        "potim": 0.10,
        "ibrion": (3 if relaxation else -1),
        "isif": 0,
        "ediffg": -0.05,
        "algo": "fast",
        "lreal": "auto",
        "nsim": 1,
        "lplane": False,
        "npar": 16,
        "nelmin": 4,
        "nsw": (800 if relaxation else 1),
        "ismear": 0,
        "sigma": 0.2,
        "ldau": "t",
        "ldauu": (2.5, 0, 0),
        "ldauj": (0, 0, 0),
        "ldaul": (2, -1, -1),
        "amix": 0.20,
        "bmix": 0.0010,
        "amix_mag": 0.80,
        "bmix_mag": 0.0010,
    }


def get_pt_on_ceo2_settings(relaxation=False):
    return {
        "command": "srun -E vasp_std",
        "lcharg": False,
        "lwave": False,
        "prec": "Normal",
        "encut": 300,
        "lmaxmix": 6,
        "icharg": 2,  # 0, #  1,   no initial CHGCAR file
        "idipol": 3,
        "ispin": 2,
        "lorbit": 11,
        "potim": 0.10,  # may increase, getting "BRIONS problems: POTIM should be increased" message
        "ibrion": (3 if relaxation else -1),
        "isif": 0,
        "ediffg": -0.05,
        "algo": "Fast",
        "lreal": "Auto",
        "nsim": 4,
        "lplane": True,
        "nelmin": 4,
        "nsw": (800 if relaxation else 1),
        "ismear": 0,
        "sigma": 0.2,
        "ldau": True,
        "ldauu": (5, 0, 0),
        "ldauj": (0, 0, 0),
        "ldaul": (3, -1, -1),
        "amix": 0.20,
        "bmix": 0.0010,
        "amix_mag": 0.80,
        "bmix_mag": 0.0010,
        # 'npar': 4,
        # 'ncore': 2,  # number of cores per band
    }


def pair_dft(element_a, element_b, dist, relaxation, get_vasp_settings):
    calc_name = "%s%s_pair_%08dpm" % tuple(
        sorted([element_a, element_b]) + [int(100 * dist)]
    )
    if calc_name in calcs:
        return calcs[calc_name]

    # Set up the atoms object at a fixed distance
    atoms = Atoms(
        [Atom(element_a, position=[0, 0, 0]), Atom(element_b, position=[0, 0, dist])]
    )
    atoms.set_cell([15, 15, 15])
    atoms.center()

    # Make the calculator
    # calc = Vasp2(
    #     atoms=atoms,
    #     directory=calc_name,
    #     label=calc_name,
    #     **get_vasp_settings(relaxation)
    # )
    calc = EMT()
    atoms.set_calculator(calc)
    dyn = SciPyFminCG(atoms)
    dyn.run(fmax=0.05)

    # Return the single-point energy
    energy = atoms.get_potential_energy()
    calcs[calc_name] = atoms
    return atoms


def single_dft(element, get_vasp_settings):
    calc_name = "%s_single" % element
    if calc_name in calcs:
        return calcs[calc_name]

    atoms = Atoms([Atom(element, position=[0, 0, 0])])
    atoms.set_cell([10, 10, 10])
    atoms.center()
    # calc = Vasp2(
    #     atoms=atoms, directory=calc_name, label=calc_name, **get_vasp_settings(False)
    # )
    calc = EMT()
    atoms.set_calculator(calc)

    energy = atoms.get_potential_energy()
    calcs[calc_name] = atoms

    return atoms


def morse_potential(r, De, re, a):
    # calculate the morse potential energy
    return De * (np.exp(-2 * a * (r - re)) - 2 * np.exp(-a * (r - re)))


def bond_calcs(element_a, element_b, get_vasp_settings):

    # Get the equilibrium bond distance/energy
    atom_a = single_dft(element_a, get_vasp_settings)
    atom_b = single_dft(element_b, get_vasp_settings)
    pair_ab = pair_dft(element_a, element_b, 3.0, True, get_vasp_settings)

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
    point_calcs = [
        pair_dft(element_a, element_b, dist, False, get_vasp_settings)
        for dist in distances
    ]
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
    return element_a, element_b, De, re, sig, distances, stretched_energies


def run_calculations(elements, **settings):
    combos = sorted(
        set([tuple(sorted(combo)) for combo in combinations(elements * 2, 2)])
    )
    print(combos)
    all_results = []
    for combo in combos:
        print("beginning", combo)
        st = time.time()
        results = list(bond_calcs(*combo, settings))
        ft = time.time()
        print(combo, "finished in %f seconds" % (ft - st))
        all_results.append(results)

    print(list(calcs.keys()))
    return all_results


# run_calculations(['Cu', 'Zn', 'O'], get_cu_on_zno_settings)
# run_calculations(["Pt", "Ce", "O"], get_pt_on_ceo2_settings)
# run_calculations(["Rh", "Ti", "O"], get_rh_on_tio2_settings)


# if __name__ == 'main':
import json

morse_data_dir = "morse_params"
if not os.path.isdir(morse_data_dir):
    os.makedirs(morse_data_dir)

pt_ag_au_morse_data = list(run_calculations(["Pt", "Ag", "Au"]))[0]
stretched_energies = pt_ag_au_morse_data.pop(-1)
distances = pt_ag_au_morse_data.pop(-1)


# with open(os.path.join(morse_data_dir, "pt_ag_au_emt_morse_data.json"), "w") as f:
#     json.dump(pt_ag_au_morse_data, f)
