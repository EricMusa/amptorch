from ase.ga.data import PrepareDB
from ase.ga.startgenerator import StartGenerator
from ase.ga.utilities import closest_distances_generator
from ase.ga.utilities import get_all_atom_types
from ase.ga.data import DataConnection
from ase.ga.population import Population
from ase.ga.standard_comparators import InteratomicDistanceComparator
from ase.ga.cutandsplicepairing import CutAndSplicePairing
from ase.ga.utilities import closest_distances_generator
from ase.ga.utilities import get_all_atom_types
from ase.ga.offspring_creator import OperationSelector
from ase.ga.standardmutations import MirrorMutation
from ase.ga.standardmutations import RattleMutation
from ase.ga.standardmutations import PermutationMutation
from ase.constraints import FixAtoms
import numpy as np
from ase.build import fcc100
from random import random
from ase.data import atomic_numbers
from ase.db import connect
from ase.formula import Formula
from ase.io import read, write
from ase.optimize.sciopt import SciPyFminCG
from ase.calculators.emt import EMT


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
# population_size = 20
# starting_population = [sg.get_new_candidate() for i in range(population_size)]
# print('created starting population of %d structures' % population_size)

############################################################


## CREATE GA STRUCTURE DATABASE AND ADD UNRELAXED STARTING POP
db_file = 'gadb.db'

# create the database to store information in
# d = PrepareDB(db_file_name=db_file,
#               simulation_cell=slab,
#               stoichiometry=atom_numbers)
# for a in starting_population:
#     d.add_unrelaxed_candidate(a)

############################################################


## DEFINE GA PARAMETERS AND RECONNECT TO GA STRUCTURE DATABASE
# Change the following three parameters to suit your needs
population_size = 20
operations = ['crossover', 'mutation', 'cross_and_mut']
probabilities = [.3, .5, .2]
n_to_test = 20

# Initialize the different components of the GA
da = DataConnection('gadb.db')
atom_numbers_to_optimize = da.get_atom_numbers_to_optimize()
n_to_optimize = len(atom_numbers_to_optimize)
slab = da.get_slab()
all_atom_types = get_all_atom_types(slab, atom_numbers_to_optimize)
blmin = closest_distances_generator(all_atom_types,
                                    ratio_of_covalent_radii=0.7)

comp = InteratomicDistanceComparator(n_top=n_to_optimize,
                                     pair_cor_cum_diff=0.015,
                                     pair_cor_max=0.7,
                                     dE=0.02,
                                     mic=False)

pairing = CutAndSplicePairing(slab, n_to_optimize, blmin)
mutations = OperationSelector([1., 1., 1.],
                              [MirrorMutation(blmin, n_to_optimize),
                               RattleMutation(blmin, n_to_optimize),
                               PermutationMutation(n_to_optimize)])
############################################################

temp_trajectory = 'temp.traj'


def relax_atoms(atoms, calc=EMT, trajectory=temp_trajectory, max_steps=25):  # 000000):
    print('relaxing atoms')
    atoms.set_calculator(calc())
    dyn = SciPyFminCG(atoms, trajectory=temp_trajectory, logfile=None)
    dyn.run(fmax=0.05, steps=max_steps)
    # capture relaxation steps for ML training material
    trajectory = read(temp_trajectory, index=':')
    with connect(mldb_file) as db:
        for image in trajectory:
            image_id = db.write(image)
            mldb_ids.append(image_id)
            # print('added image #%d to MLDB' % image_id)
        print('%d images total stored in MLDB' % len(db))



## CREATE POPULATION OBJECT AND RELAX INITIAL STRUCTURES

# connect to DB for ML Training Data
mldb_file = 'mldb.db'
mldb_ids = []

# Relax all unrelaxed structures (e.g. the starting population)
while da.get_number_of_unrelaxed_candidates() > 0:
    a = da.get_an_unrelaxed_candidate()
    relax_atoms(a)
    a.info['key_value_pairs']['raw_score'] = -a.get_potential_energy()
    da.add_relaxed_step(a)

# create the population
population = Population(data_connection=da,
                        population_size=population_size,
                        comparator=comp)

# test n_to_test new candidates
for i in range(n_to_test):
    print('Now starting configuration number {0}'.format(i))
    
    candidate = None
    attempts = 0
    while candidate is None:
        operation = np.random.choice(operations, p=probabilities)
        if operation == 'mutate':
            a = population.get_one_candidate()
            a_mut, desc = mutations.get_new_individual([a])
            if a_mut is not None:
                da.add_unrelaxed_candidate(a_mut, desc)
            candidate = a_mut
        else:
            a, b = population.get_two_candidates()
            ab, desc = pairing.get_new_individual([a, b])
            if ab is not None:
                da.add_unrelaxed_candidate(ab, desc)
                if operation == 'cross_and_mut':
                    ab, desc = mutations.get_new_individual([ab])
                    if ab is not None:
                        da.add_unrelaxed_step(ab, desc)
            candidate = ab
        
        attempts += 1
        if candidate is not None:
            print('candidate (%d) generated with %s operation after %d attempts' % (candidate.info['confid'], operation, attempts))

    # Relax the new candidate
    relax_atoms(candidate)
    candidate.info['key_value_pairs']['raw_score'] = -candidate.get_potential_energy()
    da.add_relaxed_step(candidate)
    population.update()

write('all_candidates.traj', da.get_all_relaxed_candidates())
