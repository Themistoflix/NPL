from LocalOpt.LocalOptimization import setup_local_optimization
from LocalOpt.LocalOptimization import update_atomic_features

import copy


def run_basin_hopping(start_particle, energy_calculator, environment_energies, n_hopping_attempts, n_hops):
    energy_key, local_env_calculator, local_feature_classifier, exchange_operator = setup_local_optimization(
        start_particle, energy_calculator, environment_energies)

    start_energy = start_particle.get_energy(energy_key)
    accepted_energies = [(start_energy, 0)]
    best_particle = copy.deepcopy(start_particle)
    lowest_energy = start_energy

    step = 0
    for i in range(n_hopping_attempts):
        while True:
            step += 1
            index1, index2 = exchange_operator.guided_exchange(start_particle)
            exchanged_indices = [index1, index2]

            start_particle, neighborhood = update_atomic_features(index1, index2, local_env_calculator,
                                                                  local_feature_classifier, start_particle)
            exchange_operator.update(start_particle, neighborhood, exchanged_indices)

            energy_calculator.compute_energy(start_particle)
            new_energy = start_particle.get_energy(energy_key)
            accepted_energies.append((new_energy, step))

            if new_energy < start_energy:
                start_energy = new_energy
                lowest_energy = min(lowest_energy, start_energy)
            else:
                print('Energy after local_opt: {}, lowest {}'.format(start_energy, lowest_energy))

                if lowest_energy == start_energy:
                    start_particle.swap_symbols([(index1, index2)])
                    best_particle = copy.deepcopy(start_particle)
                    best_particle.set_energy(energy_key, start_energy)

                    start_particle.swap_symbols([(index1, index2)])
                break

        for hop in range(n_hops):
            step += 1
            index1, index2 = exchange_operator.basin_hop_step(start_particle)

            exchanged_indices = [index1, index2]
            start_particle, neighborhood = update_atomic_features(index1, index2, local_env_calculator,
                                                                  local_feature_classifier, start_particle)
            exchange_operator.update(start_particle, neighborhood, exchanged_indices)

            energy_calculator.compute_energy(start_particle)
            new_energy = start_particle.get_energy(energy_key)
            accepted_energies.append((new_energy, step))

            start_energy = new_energy

    return [best_particle, accepted_energies]
