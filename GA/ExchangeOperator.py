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




