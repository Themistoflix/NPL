import copy
import numpy as np

from GA.ExchangeOperator import ExchangeOperator


def local_optimization(n_steps, beta, start_particle, local_env_calculator, feature_classifier, energy_calculator, local_energies):
    energy_key = energy_calculator.get_energy_key()

    local_env_calculator.compute_local_environments(start_particle)
    feature_classifier.compute_feature_vector(start_particle)

    energy_calculator.compute_energy(start_particle)

    old_E = start_particle.get_energy(energy_key)

    start_particle.compute_exchange_energies(local_energies, feature_classifier, start_particle.get_indices())

    best_particles = [(start_particle, 0)]
    lowest_energy = old_E

    for step in range(n_steps):
        print("Step: {0}".format(step))
        new_particle = copy.deepcopy(start_particle)

        exchanges = new_particle.get_exchange_energies_as_list()
        exchange = min(exchanges, key=lambda x: x[2])
        if exchange[2] == 0:
            break

        index_a = exchange[0]
        index_b = exchange[1]

        symbol_a = new_particle.get_symbol(index_a)
        symbol_b = new_particle.get_symbol(index_b)

        print("Symbol a: {0}".format(symbol_a))
        print("Symbol b: {0}".format(symbol_b))

        old_indices_a = start_particle.atoms.get_indices_by_symbol(symbol_a)
        old_indices_b = start_particle.atoms.get_indices_by_symbol(symbol_b)

        new_particle.atoms.swap_atoms([(index_a, index_b)])

        neighbors_a = new_particle.neighbor_list[index_a]
        neighbors_b = new_particle.neighbor_list[index_b]
        neighborhood = neighbors_a | neighbors_b | {index_a} | {index_b}
        for index in neighborhood:
            new_env = local_env_calculator.compute_local_environment(new_particle, index)
            new_particle.set_local_environment(index, new_env)
            if start_particle.get_symbol(index) == symbol_a:
                for possible_exchange in old_indices_b:
                    if possible_exchange in new_particle.exchange_energies[index]:
                        del new_particle.exchange_energies[index][possible_exchange]
            else:
                for possible_exchange in old_indices_a:
                    if index in new_particle.exchange_energies[possible_exchange]:
                        del new_particle.exchange_energies[possible_exchange][index]

        feature_classifier.compute_feature_vector(new_particle)
        new_particle.compute_exchange_energies(local_energies, feature_classifier, neighborhood)

        energy_calculator.compute_energy(new_particle)
        new_E = new_particle.get_energy(energy_key)

        delta_E = new_E - old_E
        print("Energy: {0}".format(new_E))
        print("dE: {0}".format(delta_E))

        acceptance_rate = min(1, np.exp(-beta * delta_E))
        if np.random.random() > 1 - acceptance_rate:
            old_E = new_E
            start_particle = new_particle
        else:
            break

        if new_E < lowest_energy:
            lowest_energy = new_E
            best_particles.append((new_particle, step + 1))

    return best_particles


def basin_hop(local_relaxation_steps, local_env_calculator, feature_classifier, local_energies, energy_calculator):
    def go_backwards(local_relaxation_steps):
        n_steps = min(5, len(local_relaxation_steps))
        return local_relaxation_steps[-n_steps][0]

    def perturbation(n_steps, particle, local_env_calculator, feature_classifier, local_energies):
        for step in range(n_steps):
            print("Perturbation Step: {0}".format(step))

            exchanges = particle.get_exchange_energies_as_list()
            exchanges.sort(key=lambda x: np.abs(x[2]))
            exchange = exchanges[np.random.randint(10, size=1)[0]]

            index_a = exchange[0]
            index_b = exchange[1]

            symbol_a = particle.get_symbol(index_a)
            symbol_b = particle.get_symbol(index_b)

            old_indices_a = copy.deepcopy(particle.atoms.get_indices_by_symbol(symbol_a))
            old_indices_b = copy.deepcopy(particle.atoms.get_indices_by_symbol(symbol_b))

            particle.atoms.swap_atoms([(index_a, index_b)])

            neighbors_a = particle.neighbor_list[index_a]
            neighbors_b = particle.neighbor_list[index_b]
            neighborhood = neighbors_a | neighbors_b | {index_a} | {index_b}
            for index in neighborhood:
                new_env = local_env_calculator.compute_local_environment(particle, index)
                particle.set_local_environment(index, new_env)
                if particle.get_symbol(index) == symbol_a:
                    for possible_exchange in old_indices_b:
                        if possible_exchange in particle.exchange_energies[index]:
                            del particle.exchange_energies[index][possible_exchange]
                else:
                    for possible_exchange in old_indices_a:
                        if index in particle.exchange_energies[possible_exchange]:
                            del particle.exchange_energies[possible_exchange][index]

            feature_classifier.compute_feature_vector(particle)
            particle.compute_exchange_energies(local_energies, feature_classifier, neighborhood)

        return particle

    perturbation_steps = 8
    particle = go_backwards(local_relaxation_steps)

    particle = perturbation(perturbation_steps, particle, local_env_calculator, feature_classifier, local_energies)
    energy_calculator.compute_energy(particle)
    print(particle.get_energy(energy_calculator.get_energy_key()))
    return particle


def mc_hop(beta, steps, start_particle, local_env_calculator, feature_classifier, energy_calculator):
    exchange_operator = ExchangeOperator(0.5)
    energy_key = energy_calculator.get_energy_key()

    local_env_calculator.compute_local_environments(start_particle)
    feature_classifier.compute_feature_vector(start_particle)

    energy_calculator.compute_energy(start_particle)

    old_E = start_particle.get_energy(energy_key)

    for step in range(steps):
        new_particle = exchange_operator.random_exchange(start_particle, 2)
        local_env_calculator.compute_local_environments(new_particle)
        feature_classifier.compute_feature_vector(new_particle)

        energy_calculator.compute_energy(new_particle)
        new_E = new_particle.get_energy(energy_key)

        delta_E = new_E - old_E

        acceptance_rate = min(1, np.exp(-beta * delta_E))
        if np.random.random() > 1 - acceptance_rate:
            old_E = new_E
            start_particle = new_particle

    return start_particle


def basin_hopping(n_hops, n_relaxation_steps, start_particle, local_env_calculator, feature_classifier, energy_calculator, local_energies):
    minima = []
    all_particles = []
    beta = 150

    local_env_calculator.compute_local_environments(start_particle)
    feature_classifier.compute_feature_vector(start_particle)
    #start_particle.compute_exchange_energies(local_energies, feature_classifier, start_particle.get_indices())

    local_relaxation_steps = local_optimization(n_relaxation_steps, beta, start_particle, local_env_calculator, feature_classifier, energy_calculator, local_energies)
    minima.append(local_relaxation_steps[-1][0])
    all_particles.append(local_relaxation_steps)

    for i in range(n_hops):
        particle = mc_hop(8, 150, local_relaxation_steps[-1][0], local_env_calculator, feature_classifier, energy_calculator)
        minima.append(particle)
        local_relaxation_steps = local_optimization(n_relaxation_steps, beta, particle, local_env_calculator, feature_classifier, energy_calculator, local_energies)

        all_particles.append(local_relaxation_steps)
        minima.append(local_relaxation_steps[-1][0])

    return minima, all_particles
