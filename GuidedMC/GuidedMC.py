import numpy as np
import copy

from Core.LocalEnvironmentCalculator import NeighborCountingEnvironmentCalculator
from Core.GlobalFeatureClassifier import TopologicalFeatureClassifier
from GuidedMC.GuidedExchangeOperator import GuidedExchangeOperator


def run_guided_MC(beta, steps, start_particle, energy_calculator, linear_feature_classifier, local_energies, feature_key):
    symbols = start_particle.get_symbols()

    local_env_calculator = NeighborCountingEnvironmentCalculator(symbols)
    topological_feature_classifier = TopologicalFeatureClassifier(symbols)

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

    for i in range(1, steps + 1):
        old_atom_features = copy.deepcopy(start_particle.get_atom_features(feature_key))
        old_environments = copy.deepcopy(start_particle.get_local_environemts())

        exchanges = exchange_operator.guided_exchange(start_particle)

        neighborhood = set()
        for exchange in exchanges:
            index1 = exchange[0]
            index2 = exchange[1]

            neighborhood.add(index1)
            neighborhood.add(index2)
            neighborhood.union(start_particle.neighbor_list[index1])
            neighborhood.union(start_particle.neighbor_list[index2])

        for index in neighborhood:
            local_env_calculator.compute_local_environment(start_particle, index)
            linear_feature_classifier.compute_atom_feature(index)

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

            if new_E < lowest_energy:
                lowest_energy = new_E
                best_particles.append((start_particle, i))
        else:
            start_particle.set_local_environments(old_environments)
            start_particle.set_atom_features(old_atom_features, feature_key)

    best_particles.append((best_particles[-1][0], steps))

    return best_particles