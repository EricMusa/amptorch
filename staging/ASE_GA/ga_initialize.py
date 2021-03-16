from ase.ga.data import PrepareDB
from ase.ga.startgenerator import StartGenerator
from ase.ga.utilities import closest_distances_generator
from ase.ga.utilities import get_all_atom_types
from ase.constraints import FixAtoms
from ase.build import fcc100
from ase.data import atomic_numbers
from ase.formula import Formula
import numpy as np
import os


def init_slab_cluster_ga(cluster_composition, db_file='gadb.db', ratio_of_covalent_radii=0.7):
    slab = fcc100('Au', size=(8, 8, 4), vacuum=12.0, orthogonal=True)
    slab.pbc[2] = True
    slab.set_constraint(FixAtoms(indices=[i for i, atom in enumerate(slab) if atom.position[2] < 13.]))
    print('slab created:', slab.get_chemical_formula())
    pos = slab.get_positions()
    cell = slab.get_cell()

    a, b, c = cell[0, 0], cell[1, 1], cell[2, 2]
    p0 = np.array([a * .25, b * .25, max(pos[:, 2]) + 2.])
    v1 = np.array([a, 0, 0]) * 0.5
    v2 = np.array([0, b, 0]) * 0.5
    v3 = np.array([0, 0, c]) * 0.25

    # cluster_composition = 'Pt8Ag8'
    cluster_formula = Formula(cluster_composition)
    print('cluster composition:', cluster_formula)
    atom_numbers = []
    for element, count in cluster_formula.count().items():
        atom_numbers += [atomic_numbers[element]] * count

    unique_atom_types = get_all_atom_types(slab, atom_numbers)
    blmin = closest_distances_generator(atom_numbers=unique_atom_types,
                                        ratio_of_covalent_radii=ratio_of_covalent_radii)
    
    sg = StartGenerator(slab, atom_numbers, blmin,
                        box_to_place_in=[p0, [v1, v2, v3]])
    print('StartGenerator created, returning')
    if not os.path.isfile(db_file):
        print('creating DB file')
        d = PrepareDB(db_file_name=db_file,
                    simulation_cell=slab,
                    stoichiometry=atom_numbers)
    return sg
