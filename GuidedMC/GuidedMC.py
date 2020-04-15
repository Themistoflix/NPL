import numpy as np
import copy

from Core.LocalEnvironmentCalculator import NeighborCountingEnvironmentCalculator
from Core.GlobalFeatureClassifier import TopologicalFeatureClassifier2
from GuidedMC.GuidedExchangeOperator import GuidedExchangeOperator
from GuidedMC.GuidedExchangeOperator import RandomExchangeOperator


def run_guided_MC(beta, steps, start_particle, energy_calculator, linear_feature_classifier, local_energies, feature_key):
    symbols = start_particle.get_symbols()

    local_env_calculator = NeighborCountingEnvironmentCalculator(symbols)
    topological_feature_classifier = TopologicalFeatureClassifier2(symbols)

    energy_key = energy_calculator.get_energy_key()

    local_env_calculator.compute_local_environments(start_particle)
    topological_feature_classifier.compute_feature_vector(start_particle)

    linear_feature_classifier.compute_feature_vector(start_particle)
    energy_calculator.compute_energy(start_particle)

    exchange_operator = GuidedExchangeOperator(local_energies, 0.5, feature_key)
    exchange_operator.bind_particle(start_particle)

    old_E = start_particle.get_energy(energy_key)

    best_particles = [(start_particle, 0)]
    lowest_energy = old_E

    #old_atom_features = copy.deepcopy(start_particle.get_atom_features(feature_key))
    #old_environments = copy.deepcopy(start_particle.get_local_environments())
    for i in range(1, steps + 1):
        old_atom_features = copy.deepcopy(start_particle.get_atom_features(feature_key))
        old_environments = copy.deepcopy(start_particle.get_local_environments())

        exchanges = exchange_operator.guided_exchange(start_particle)

        exchanged_indices = []
        neighborhood = set()
        for exchange in exchanges:
            index1 = exchange[0]
            index2 = exchange[1]

            #print("Index 1: {0} symbol: {1}, Index 2: {2} symbol: {3}".format(index1, start_particle.get_symbol(index1), index2, start_particle.get_symbol(index2)))

            exchanged_indices.append(index1)
            exchanged_indices.append(index2)

            neighborhood.add(index1)
            neighborhood.add(index2)
            neighborhood = neighborhood.union(start_particle.neighbor_list[index1])
            neighborhood = neighborhood.union(start_particle.neighbor_list[index2])

        for index in neighborhood:
            local_env_calculator.compute_local_environment(start_particle, index)
            linear_feature_classifier.compute_atom_feature(start_particle, index)

        topological_feature_classifier.compute_feature_vector(start_particle)
        linear_feature_classifier.compute_feature_vector(start_particle, recompute_atom_features=False)

        energy_calculator.compute_energy(start_particle)
        new_E = start_particle.get_energy(energy_key)

        delta_E = new_E - old_E

        acceptance_rate = min(1, np.exp(-beta * delta_E))
        if np.random.random() > 1 - acceptance_rate:
            old_E = new_E
            exchange_operator.reset_index()
            print("Step: {0} New E: {1}".format(i, new_E))
            exchange_operator.update(start_particle, neighborhood, exchanged_indices)
            print(np.dot(start_particle.get_feature_vector('TEC'), local_energies))
            #old_atom_features = copy.deepcopy(start_particle.get_atom_features(feature_key))
            #old_environments = copy.deepcopy(start_particle.get_local_environments())

            if new_E < lowest_energy:
                lowest_energy = new_E
                best_particles.append((start_particle, i))
        else:
            start_particle.atoms.swap_atoms(exchanges)
            start_particle.set_local_environments(old_environments)
            start_particle.set_atom_features(old_atom_features, feature_key)

    best_particles.append((best_particles[-1][0], steps))

    return best_particles


def run_normal_MC(beta, steps, start_particle, energy_calculator, linear_feature_classifier):
    symbols = start_particle.get_symbols()

    local_env_calculator = NeighborCountingEnvironmentCalculator(symbols)
    topological_feature_classifier = TopologicalFeatureClassifier2(symbols)

    energy_key = energy_calculator.get_energy_key()

    local_env_calculator.compute_local_environments(start_particle)
    topological_feature_classifier.compute_feature_vector(start_particle)

    linear_feature_classifier.compute_feature_vector(start_particle)
    energy_calculator.compute_energy(start_particle)

    exchange_operator = RandomExchangeOperator(0.5)
    exchange_operator.bind_particle(start_particle)

    old_E = start_particle.get_energy(energy_key)

    best_particles = [(start_particle, 0)]
    lowest_energy = old_E

    old_environments = copy.deepcopy(start_particle.get_local_environments())
    for i in range(1, steps + 1):
        exchanges = exchange_operator.random_exchange(start_particle)

        exchanged_indices = []
        neighborhood = set()
        for exchange in exchanges:
            index1 = exchange[0]
            index2 = exchange[1]

            #print("Index 1: {0} symbol: {1}, Index 2: {2} symbol: {3}".format(index1, start_particle.get_symbol(index1), index2, start_particle.get_symbol(index2)))

            exchanged_indices.append(index1)
            exchanged_indices.append(index2)

            neighborhood.add(index1)
            neighborhood.add(index2)
            neighborhood = neighborhood.union(start_particle.neighbor_list[index1])
            neighborhood = neighborhood.union(start_particle.neighbor_list[index2])

        for index in neighborhood:
            local_env_calculator.compute_local_environment(start_particle, index)
            linear_feature_classifier.compute_atom_feature(start_particle, index)

        topological_feature_classifier.compute_feature_vector(start_particle)
        linear_feature_classifier.compute_feature_vector(start_particle, recompute_atom_features=False)

        energy_calculator.compute_energy(start_particle)
        new_E = start_particle.get_energy(energy_key)

        delta_E = new_E - old_E

        acceptance_rate = min(1, np.exp(-beta * delta_E))
        if np.random.random() > 1 - acceptance_rate:
            old_E = new_E
            old_environments = copy.deepcopy(start_particle.get_local_environments())
            print("Step: {0} New E: {1}".format(i, new_E))


            if new_E < lowest_energy:
                lowest_energy = new_E
                best_particles.append((start_particle, i))
        else:
            start_particle.atoms.swap_atoms(exchanges)
            start_particle.set_local_environments(old_environments)

    best_particles.append((best_particles[-1][0], steps))

    return best_particles
