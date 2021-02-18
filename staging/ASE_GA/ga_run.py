import numpy as np
from ase.io import read, write
from ase.optimize.sciopt import SciPyFminCG
from ase.calculators.emt import EMT
from ase.db import connect
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


## DEFINE GA PARAMETERS AND RECONNECT TO GA STRUCTURE DATABASE
# Change the following three parameters to suit your needs
population_size = 20
operations = ['crossover', 'mutation', 'cross_and_mut']
probabilities = [.3, .5, .2]
n_to_test = 20

# Initialize the different components of the GA
da = DataConnection('gadb2.db')
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
mldb_file = 'mldb2.db'
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

write('all_candidates2.traj', da.get_all_relaxed_candidates())
