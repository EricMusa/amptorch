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


def init_ga_connection(db_file='gadb.db', population_size=20, ratio_of_covalent_radii=0.7):
    print('connecting to %s' % db_file)
    da = DataConnection(db_file)
    atom_numbers_to_optimize = da.get_atom_numbers_to_optimize()
    n_to_optimize = len(atom_numbers_to_optimize)
    slab = da.get_slab()
    all_atom_types = get_all_atom_types(slab, atom_numbers_to_optimize)
    blmin = closest_distances_generator(all_atom_types,
                                        ratio_of_covalent_radii=ratio_of_covalent_radii)      
    data = {
        'atom_numbers_to_optimize': atom_numbers_to_optimize,
        'n_to_optimize': n_to_optimize,
        'slab': slab,
        'all_atom_types': all_atom_types,
        'blmin': blmin,
    }                     
    print('connected, data extracted')

    comp = InteratomicDistanceComparator(n_top=n_to_optimize,
                                        pair_cor_cum_diff=0.015,
                                        pair_cor_max=0.7,
                                        dE=0.02,
                                        mic=False)
    print('comparator created')

    pairing = CutAndSplicePairing(slab, n_to_optimize, blmin)
    print('splice pairing created')

    mutations = OperationSelector([1., 1., 1.],
                                [MirrorMutation(blmin, n_to_optimize),
                                RattleMutation(blmin, n_to_optimize),
                                PermutationMutation(n_to_optimize)]) 
    print('mutations created')

    population = Population(data_connection=da,
                            population_size=population_size,
                            comparator=comp)
    print('population created')

    return population, da, comp, pairing, mutations, data


def relax_atoms_emt(atoms, trajectory='temp.traj', logfile=None, fmax=0.05, max_steps=25, db_file=None):
    print('relaxing atoms')
    atoms.set_calculator(EMT())
    dyn = SciPyFminCG(atoms, trajectory=trajectory, logfile=logfile)
    dyn.run(fmax=fmax, steps=max_steps)
    trajectory = read(trajectory, index=':')
    if db_file:
        with connect(db_file) as db:
            for image in trajectory:
                image_id = db.write(image)
            print('%d images total stored in MLDB' % len(db))


def relax_unrelaxed_candidates(data_connection, relax_func=relax_atoms_emt, **relax_kwargs):
    while data_connection.get_number_of_unrelaxed_candidates() > 0:
        a = data_connection.get_an_unrelaxed_candidate()
        relax_func(a, **relax_kwargs)
        a.info['key_value_pairs']['raw_score'] = -a.get_potential_energy()
        data_connection.add_relaxed_step(a)


def generate_relaxed_candidate(population, da, mutations, probabilities, 
    cross_pairing, relax_func=relax_atoms_emt, **relax_kwargs):
    candidate = None
    attempts = 0
    while candidate is None:
        operation = np.random.choice(['crossover', 'mutation', 'cross_and_mut'], p=probabilities)
        if operation == 'mutate':
            a = population.get_one_candidate()
            a_mut, desc = mutations.get_new_individual([a])
            if a_mut is not None:
                da.add_unrelaxed_candidate(a_mut, desc)
            candidate = a_mut
        else:
            a, b = population.get_two_candidates()
            ab, desc = cross_pairing.get_new_individual([a, b])
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
        relax_func(candidate, **relax_kwargs)
        candidate.info['key_value_pairs']['raw_score'] = -candidate.get_potential_energy()
        da.add_relaxed_step(candidate)
        population.update()

