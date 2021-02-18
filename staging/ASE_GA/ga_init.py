from ase.ga.data import PrepareDB
from ase.ga.startgenerator import StartGenerator
from ase.ga.utilities import closest_distances_generator
from ase.ga.utilities import get_all_atom_types
from ase.constraints import FixAtoms
from ase.build import fcc100
from ase.data import atomic_numbers
from ase.formula import Formula
import numpy as np


## SET UP THE SURFACE AND DEFINE THE SPACE TO BE EXPLORED ##
# create the surface
slab = fcc100('Au', size=(8, 8, 4), vacuum=10.0, orthogonal=True)
slab.set_constraint(FixAtoms(indices=[i for i, atom in enumerate(slab) if atom.position[2] < 13.]))
print('using %s surface' % slab.get_chemical_formula())

# define the volume in which the adsorbed cluster is optimized
# the volume is defined by a corner position (p0)
# and three spanning vectors (v1, v2, v3)
pos = slab.get_positions()
cell = slab.get_cell()

a, b, c = cell[0, 0], cell[1, 1], cell[2, 2]
p0 = np.array([a * .25, b * .25, max(pos[:, 2]) + 2.])
v1 = np.array([a, 0, 0]) * 0.5
v2 = np.array([0, b, 0]) * 0.5
v3 = np.array([0, 0, c]) * 0.25

# Define the composition of the atoms to optimize
cluster_composition = 'Pt8Ag8'
cluster_formula = Formula(cluster_composition)
atom_numbers = []
for element, count in cluster_formula.count().items():
    atom_numbers += [atomic_numbers[element]] * count
print('nanocluster composition: %s' % cluster_composition)
############################################################

## DO INITIALIZATION AND CREATE UNRELAXED STARTING POPULATION
# define the closest distance two atoms of a given species can be to each other
unique_atom_types = get_all_atom_types(slab, atom_numbers)
blmin = closest_distances_generator(atom_numbers=unique_atom_types,
                                    ratio_of_covalent_radii=0.7)
print('minimum bond lengths', blmin)

# create the starting population
sg = StartGenerator(slab, atom_numbers, blmin,
                    box_to_place_in=[p0, [v1, v2, v3]])

# generate the starting population
population_size = 20
starting_population = [sg.get_new_candidate() for i in range(population_size)]
print('created starting population of %d structures' % population_size)

############################################################


## CREATE GA STRUCTURE DATABASE AND ADD UNRELAXED STARTING POP
db_file = 'gadb2.db'

# create the database to store information in
d = PrepareDB(db_file_name=db_file,
              simulation_cell=slab,
              stoichiometry=atom_numbers)
for a in starting_population:
    d.add_unrelaxed_candidate(a)

############################################################
