import numpy as np
import copy


class ExchangeOperator:
    def __init__(self, p):
        self.p = p
        self.index = 0

    def random_exchange(self, particle):
        new_particle = copy.deepcopy(particle)
        if new_particle.is_pure():
            print("Pure particle! No permutation possible")
            return new_particle

        symbols = new_particle.atoms.get_symbols()
        symbol1 = symbols[0]
        symbol2 = symbols[1]

        max_permutations = min(len(new_particle.atoms.get_indices_by_symbol(symbol1)), len(new_particle.atoms.get_indices_by_symbol(symbol2)))
        n_permutations = min(self.draw_from_geometric_distribution(self.p), max_permutations)

        symbol1_indices = np.random.choice(new_particle.atoms.get_indices_by_symbol(symbol1), n_permutations, replace=False)
        symbol2_indices = np.random.choice(new_particle.atoms.get_indices_by_symbol(symbol2), n_permutations, replace=False)

        new_particle.atoms.swap_atoms(zip(symbol1_indices, symbol2_indices))

        return new_particle

    def draw_from_geometric_distribution(self, p):
        return np.random.geometric(p=p, size=1)[0]

    def guided_exchange(self, particle, local_energies, feature_key):
        def env_feature(x):
            if x >= len(local_energies)/2:
                return int(x - len(local_energies)/2)
            else:
                return x

        new_particle = copy.deepcopy(particle)
        if new_particle.is_pure():
            print("Pure particle! No permutation possible")
            return new_particle

        symbols = sorted(new_particle.atoms.get_symbols())
        symbol1 = symbols[0]
        symbol2 = symbols[1]

        # compute differences of species in same atomic environment
        n_envs = int(len(local_energies) / 2)
        env_differences = [local_energies[i] - local_energies[i + n_envs] for i in range(n_envs)]

        atom_features = particle.get_atom_features(feature_key)

        symbol1_indices = new_particle.get_indices_by_symbol(symbol1)
        np.random.shuffle(symbol1_indices)
        symbol1_indices.sort(key=lambda x: env_differences[env_feature(atom_features[x])] if env_differences[env_feature(atom_features[x])] > 0 else 0, reverse=True)

        symbol2_indices = new_particle.get_indices_by_symbol(symbol2)
        np.random.shuffle(symbol2_indices)
        symbol2_indices.sort(key=lambda x: env_differences[env_feature(atom_features[x])] if env_differences[env_feature(atom_features[x])] < 0 else 0)

        symbol1_index = symbol1_indices[self.index % len(symbol1_indices)]
        symbol2_index = symbol2_indices[self.index % len(symbol2_indices)]

        new_particle.atoms.swap_atoms([(symbol1_index, symbol2_index)])
        self.index += 1

        return new_particle

    def reset(self):
        self.index = 0

    def gradient_descent(self, particle, local_energies, feature_key, local_feature_classifier):
        current_features = particle.get_atom_features(feature_key)

        def compute_neighborhood_energy(index, local_energies):
            energy = local_energies[current_features[index]]
            neighbors = particle.neighbor_list[index]
            for neighbor in neighbors:
                energy += local_energies[current_features[neighbor]]
            return energy

        def compute_exchange_energies(symbol_from, symbol_to):
            indices_with_symbol = copy.deepcopy(new_particle.get_indices_by_symbol(symbol_from))
            indices_with_symbol.sort()
            exchange_neighborhood_energies = dict()

            for index in indices_with_symbol:
                old_neighborhood_energy = compute_neighborhood_energy(index, local_energies)
                new_particle.atoms.transform_atoms([(index, symbol_to)])

                new_neighborhood_energy = 0

                neighbors = new_particle.neighbor_list[index]
                for atom_index in neighbors | {index}:
                    new_feature = local_feature_classifier.predict_atom_feature(new_particle, atom_index, True)
                    new_neighborhood_energy += local_energies[new_feature]

                exchange_neighborhood_energies[index] = new_neighborhood_energy - old_neighborhood_energy
                new_particle.atoms.transform_atoms([(index, symbol_from)])

            return exchange_neighborhood_energies

        new_particle = copy.deepcopy(particle)
        if new_particle.is_pure():
            print("Pure particle! No permutation possible")
            return new_particle

        symbols = sorted(new_particle.atoms.get_symbols())
        symbol_a = symbols[0]
        symbol_b = symbols[1]

        # exchange energies symbol_a atoms with symbol_b atoms
        exchange_energies_to_b = compute_exchange_energies(symbol_a, symbol_b)
        exchange_energies_to_a = compute_exchange_energies(symbol_b, symbol_a)


        # sort indices according to maximal energy gain = minimal exchange energy
        best_indices_a = [index for index, energy in sorted(exchange_energies_to_b.items(), key=lambda x: x[1])]
        best_indices_b = [index for index, energy in sorted(exchange_energies_to_a.items(), key=lambda x: x[1])]

        print("{0}{1}{2}".format(exchange_energies_to_a[best_indices_b[0]], exchange_energies_to_a[best_indices_b[1]], exchange_energies_to_a[best_indices_b[2]]))
        print("{0}{1}{2}".format(exchange_energies_to_b[best_indices_a[0]], exchange_energies_to_b[best_indices_a[1]],
                                 exchange_energies_to_b[best_indices_a[2]]))

        n_atoms_symbol_a = len(best_indices_a)
        n_atoms_symbol_b = len(best_indices_b)

        #index_atom_a = min(self.draw_from_geometric_distribution(self.p) - 1, n_atoms_symbol_a)
        #index_atom_b = min(self.draw_from_geometric_distribution(self.p) - 1, n_atoms_symbol_b)

        #print(index_atom_a)
        #print(index_atom_b)

        new_particle.atoms.swap_atoms([(best_indices_a[self.index % n_atoms_symbol_a], best_indices_b[self.index % n_atoms_symbol_b])])
        self.index += 1

        return new_particle






