import numpy as np
from GA import CutAndSpliceOperator
from GA import ExchangeOperator
from GA import MutationOperator

import Core.Profiler


@Core.Profiler.profile
def run_single_particle_GA(start_population, unsuccessful_gens_for_convergence, energy_calculator, local_env_calculator, local_feature_classifier):
    unsuccessful_gens = 0
    energy_key = energy_calculator.get_energy_key()
    symbols = start_population[0].get_symbols()

    population_size = len(start_population)
    n_parents = int(0.3*population_size)

    for p in start_population:
        local_env_calculator.compute_local_environments(p)
        local_feature_classifier.compute_feature_vector(p)
        energy_calculator.compute_energy(p)

    mutation_operator = MutationOperator.MutationOperator(0.5, symbols)
    exchange_operator = ExchangeOperator.ExchangeOperator(0.5)
    cut_and_splice_operator = CutAndSpliceOperator.CutAndSpliceOperator(0, 10)

    cur_population = start_population
    generation = 0
    best_energies = []
    while unsuccessful_gens < unsuccessful_gens_for_convergence:
        print("Generation: {}".format(generation))
        cur_population.sort(key=lambda x: x.get_energy(energy_key))
        best_energies.append((cur_population[0].get_energy(energy_key), generation))
        generation += 1

        parents = cur_population[:n_parents]

        for i in range(population_size):
            p = np.random.rand()
            if p < 0.2:
                parent1, parent2 = np.random.choice(parents, 2, replace=False)
                new_offspring = cut_and_splice_operator.cut_and_splice(parent1, parent2)
            else:
                parent = np.random.choice(parents, 1)[0]
                new_offspring = exchange_operator.random_exchange(parent)
            # else:
            #   parent = np.random.choice(parents, 1)[0]
            #    new_offspring = mutation_operator.random_mutation(parent)

            local_env_calculator.compute_local_environments(new_offspring)
            local_feature_classifier.compute_feature_vector(new_offspring)
            energy_calculator.compute_energy(new_offspring)

            cur_population.append(new_offspring)

        if generation > 10:
            break

        cur_population.sort(key=lambda x: x.get_energy(energy_key))
        cur_population = cur_population[:population_size]

        if cur_population[0].get_energy(energy_key) == best_energies[-1][0]:
            unsuccessful_gens += 1
        else:
            unsuccessful_gens = 0
            print("New best energy: {}".format(cur_population[0].get_energy(energy_key)))

    cur_population.sort(key=lambda x: x.get_energy(energy_key))

    return best_energies, generation, cur_population[0]