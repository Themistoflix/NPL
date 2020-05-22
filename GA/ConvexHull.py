import numpy as np
from GA import CutAndSpliceOperator
from GA import ExchangeOperator
from GA import MutationOperator

import copy
import pickle

def locate_convex_hull(start_population, unsuccessful_gens_for_convergence, energy_calculator, local_env_calculator, local_feature_classifier):
    gens_since_last_success = 0
    symbols = ['Pt', 'Au']

    mutation_operator = MutationOperator.MutationOperator(0.5, symbols)
    exchange_operator = ExchangeOperator.ExchangeOperator(0.5)
    cut_and_splice_operator = CutAndSpliceOperator.CutAndSpliceOperator(0, 10)

    population = copy.deepcopy(start_population)
    for p in population.population.values():
        local_env_calculator.compute_local_environments(p)
        local_feature_classifier.compute_feature_vector(p)
        energy_calculator.compute_energy(p)

    energy_log = []
    energy_log.append(population.get_convex_hull())

    cur_generation = 0
    while gens_since_last_success < unsuccessful_gens_for_convergence:
        cur_generation += 1
        print("Current generation: {0}".format(cur_generation))
        print(" ")
        print(['-' * 40])
        print(" ")

        if cur_generation % 200 == 0:
            energy_log.append(population.get_convex_hull())
            pickle.dump(energy_log, open("energy_log.pkl", 'wb'))

        # choose the new particle from a priority region
        # priority_compositions = determine_priority_compositions(energy_log, 0.9)
        # print("Priority: {0}".format(priority_compositions))

        p = np.random.random()
        if p < 0.4:
            # choose two parents for cut and splice
            print("Cut and Splice")
            parent1, parent2 = population.tournament_selection(2, 5)
            new_particle = cut_and_splice_operator.cut_and_splice(parent1, parent2, False)

                # if new_particle.get_stoichiometry()['Pt'] in priority_compositions:
                #    break

        elif p < 0.8:
            # random exchange
            print("random exchange")
            parent = population.random_selection(1)[0]
            new_particle = exchange_operator.random_exchange(parent)

                #if new_particle.get_stoichiometry()['Pt'] in priority_compositions:
                #    break

        else:
            # random mutation
            print("random mutation")
            parent = population.gaussian_tournament(1, 5)[0]
            new_particle = mutation_operator.random_mutation(parent)

                #if new_particle.get_stoichiometry()['Pt'] in priority_compositions:
                #    break

        # check that it is not a pure particle
        if new_particle.is_pure():
            continue

        local_env_calculator.compute_local_environments(new_particle)
        local_feature_classifier.compute_feature_vector(new_particle)
        energy_calculator.compute_energy(new_particle)
        print("New Energy: {}".format(new_particle.get_energy('BRR')))

        # add new offspring to population
        successfull_offspring = False
        niche = population.get_niche(new_particle)

        if new_particle.get_energy('BRR') < population[niche].get_energy('BRR'):
            print("success")
            successfull_offspring = True
            population.add_offspring(new_particle)

        # reset counters and log energy
        if successfull_offspring:
            gens_since_last_success = 0
        else:
            gens_since_last_success += 1

    return [population, energy_log]