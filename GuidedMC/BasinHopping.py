import numpy as np
import Core.Profiler
import copy

from Core.LocalEnvironmentCalculator import NeighborCountingEnvironmentCalculator
from GuidedMC.GuidedExchangeOperator import GuidedExchangeOperator


def run_basin_hopping(start_particle, energy_calculator, local_feature_classifier, total_energies, n_hopping_attempts, n_hops):
    symbols = start_particle.get_all_symbols()
    local_env_calculator = NeighborCountingEnvironmentCalculator(symbols)
    energy_key = energy_calculator.get_energy_key()

    local_env_calculator.compute_local_environments(start_particle)

    local_feature_classifier.compute_feature_vector(start_particle)
    feature_key = local_feature_classifier.get_feature_key()
    energy_calculator.compute_energy(start_particle)

    local_energies = energy_calculator.get_coefficients()

    exchange_operator = GuidedExchangeOperator(local_energies, total_energies, feature_key)
    exchange_operator.bind_particle(start_particle)

    old_E = start_particle.get_energy(energy_key)
    lowest_E = old_E
    best_particle = copy.deepcopy(start_particle.get_as_dictionary(True))
    accepted_energies = [(old_E, 0)]

    step = 0
    for i in range(n_hopping_attempts):
        while True:
            exchanges = exchange_operator.guided_exchange(start_particle)
            step += 1

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
                local_feature_classifier.compute_atom_feature(start_particle, index)

            local_feature_classifier.compute_feature_vector(start_particle, recompute_atom_features=False)

            energy_calculator.compute_energy(start_particle)
            new_E = start_particle.get_energy(energy_key)


            exchange_operator.reset_index()
            exchange_operator.update(start_particle, neighborhood, exchanged_indices)

            if new_E < old_E:
                old_E = new_E
                accepted_energies.append((new_E, step))
            else:
                if old_E < lowest_E:
                    start_particle.atoms.swap_atoms(exchanges)
                    print('lowest E so far: {}, now: {}'.format(lowest_E, old_E))
                    lowest_E = old_E
                    best_particle = copy.deepcopy(start_particle.get_as_dictionary())
                    best_particle['energies'][energy_key] = old_E

                    start_particle.atoms.swap_atoms(exchanges)
                break

        for hop in range(n_hops):
            exchanges = exchange_operator.basin_hop_step(start_particle)
            step += 1

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
                local_feature_classifier.compute_atom_feature(start_particle, index)

            local_feature_classifier.compute_feature_vector(start_particle, recompute_atom_features=False)

            energy_calculator.compute_energy(start_particle)
            new_E = start_particle.get_energy(energy_key)
            old_E = new_E

            exchange_operator.reset_index()
            exchange_operator.update(start_particle, neighborhood, exchanged_indices)

    accepted_energies.append((accepted_energies[-1][0], step))

    return [accepted_energies, best_particle]