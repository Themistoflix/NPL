import numpy as np
import copy

from Core.LocalEnvironmentCalculator import NeighborCountingEnvironmentCalculator
from GuidedMC.GuidedExchangeOperator import GuidedExchangeOperator
from GuidedMC.GuidedExchangeOperator import RandomExchangeOperator


def run_guided_MC(beta, steps, start_particle, energy_calculator, linear_feature_classifier, local_energies, feature_key, stochastic=False):
    symbols = start_particle.get_symbols()
    local_env_calculator = NeighborCountingEnvironmentCalculator(symbols)
    energy_key = energy_calculator.get_energy_key()

    local_env_calculator.compute_local_environments(start_particle)

    linear_feature_classifier.compute_feature_vector(start_particle)
    energy_calculator.compute_energy(start_particle)

    exchange_operator = GuidedExchangeOperator(local_energies, 0.5, feature_key)
    exchange_operator.bind_particle(start_particle)

    old_E = start_particle.get_energy(energy_key)
    lowest_energy = old_E
    #best_particles = [(copy.deepcopy(start_particle.get_as_dictionary(True)), 0)]
    best_energies = [(lowest_energy, 0)]
    for i in range(1, steps + 1):
        if stochastic is False:
            exchanges = exchange_operator.guided_exchange(start_particle)
        else:
            exchanges = exchange_operator.stochastic_guided_exchange(start_particle)

        exchanged_indices = []
        neighborhood = set()
        for exchange in exchanges:
            index1 = exchange[0]
            index2 = exchange[1]

            exchanged_indices.append(index1)
            exchanged_indices.append(index2)

            neighborhood.add(index1)
            neighborhood.add(index2)
            neighborhood = neighborhood.union(start_particle.neighbor_list[index1])
            neighborhood = neighborhood.union(start_particle.neighbor_list[index2])

        for index in neighborhood:
            local_env_calculator.compute_local_environment(start_particle, index)
            linear_feature_classifier.compute_atom_feature(start_particle, index)

        linear_feature_classifier.compute_feature_vector(start_particle, recompute_atom_features=False)

        energy_calculator.compute_energy(start_particle)
        new_E = start_particle.get_energy(energy_key)

        delta_E = new_E - old_E


        acceptance_rate = min(1, np.exp(-beta * delta_E))
        if np.random.random() > 1 - acceptance_rate:
            old_E = new_E
            exchange_operator.reset_index()
            best_energies.append((new_E, i))

            exchange_operator.update(start_particle, neighborhood, exchanged_indices)
            if new_E < lowest_energy:
                lowest_energy = new_E
                #best_particles.append((copy.deepcopy(start_particle.get_as_dictionary(True)), i))
        else:
            start_particle.atoms.swap_atoms(exchanges)
            for index in neighborhood:
                local_env_calculator.compute_local_environment(start_particle, index)
                linear_feature_classifier.compute_atom_feature(start_particle, index)

    best_energies.append((best_energies[-1][0], steps))
    #best_particles.append((best_particles[-1][0], steps))

    #return best_particles
    return best_energies

def run_normal_MC(beta, steps, start_particle, energy_calculator, linear_feature_classifier):
    symbols = start_particle.get_symbols()

    local_env_calculator = NeighborCountingEnvironmentCalculator(symbols)

    energy_key = energy_calculator.get_energy_key()

    local_env_calculator.compute_local_environments(start_particle)

    linear_feature_classifier.compute_feature_vector(start_particle)
    energy_calculator.compute_energy(start_particle)

    exchange_operator = RandomExchangeOperator(0.5)
    exchange_operator.bind_particle(start_particle)

    old_E = start_particle.get_energy(energy_key)
    lowest_energy = old_E
    best_particles = [(copy.deepcopy(start_particle.get_as_dictionary(True)), 0)]

    for i in range(1, steps + 1):
        exchanges = exchange_operator.random_exchange(start_particle)

        exchanged_indices = []
        neighborhood = set()
        for exchange in exchanges:
            index1 = exchange[0]
            index2 = exchange[1]

            exchanged_indices.append(index1)
            exchanged_indices.append(index2)

            neighborhood.add(index1)
            neighborhood.add(index2)
            neighborhood = neighborhood.union(start_particle.neighbor_list[index1])
            neighborhood = neighborhood.union(start_particle.neighbor_list[index2])

        for index in neighborhood:
            local_env_calculator.compute_local_environment(start_particle, index)
            linear_feature_classifier.compute_atom_feature(start_particle, index)

        linear_feature_classifier.compute_feature_vector(start_particle, recompute_atom_features=False)

        energy_calculator.compute_energy(start_particle)
        new_E = start_particle.get_energy(energy_key)

        delta_E = new_E - old_E

        acceptance_rate = min(1, np.exp(-beta * delta_E))
        if np.random.random() > 1 - acceptance_rate:
            old_E = new_E

            if new_E < lowest_energy:
                lowest_energy = new_E
                best_particles.append((copy.deepcopy(start_particle.get_as_dictionary(True)), i))
        else:
            start_particle.atoms.swap_atoms(exchanges)
            for index in neighborhood:
                local_env_calculator.compute_local_environment(start_particle, index)
                linear_feature_classifier.compute_atom_feature(start_particle, index)

    best_particles.append((best_particles[-1][0], steps))

    return best_particles
