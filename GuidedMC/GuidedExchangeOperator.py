import numpy as np
from sortedcontainers import SortedDict


class GuidedExchangeOperator:
    def __init__(self, local_energies, p_geometric, feature_key):
        self.local_energies = local_energies
        self.n_envs = int(len(local_energies)/2)
        self.env_energy_differences = [local_energies[i] - local_energies[i + self.n_envs] for i in range(self.n_envs)]

        self.feature_key = feature_key

        self.symbol1_indices = SortedDict()
        self.symbol2_indices = SortedDict()

        self.index = 0
        self.n_symbol1_atoms = 0
        self.n_symbol2_atoms = 0

        self.max_exchanges = 0
        self.p_geometric = p_geometric

    def env_from_feature(self, x):
        return x % self.n_envs

    def guided_exchange(self, particle):
        symbol1_exchange = self.symbol1_indices.peekitem(self.index % self.n_symbol1_atoms)
        symbol1_index = symbol1_exchange[0]
        symbol1_energy = symbol1_exchange[1]

        symbol2_exchange = self.symbol2_indices.peekitem(self.index % self.n_symbol2_atoms)
        symbol2_index = symbol2_exchange[0]
        symbol2_energy = symbol2_exchange[1]

        expected_energy_gain = symbol1_energy + symbol2_energy
        self.index += 1

        if expected_energy_gain < 0:
            particle.atoms.swap_atoms([(symbol1_index, symbol2_index)])
            return [(symbol1_index, symbol2_index)]
        else:
            n_exchanges = min(np.random.geometric(p=self.p_geometric, size=1)[0], self.max_exchanges)
            symbol1_indices = np.random.choice(self.symbol1_indices.keys(), n_exchanges, replace=False)
            symbol2_indices = np.random.choice(self.symbol2_indices.keys(), n_exchanges, replace=False)

            particle.atoms.swap_atoms(zip(symbol1_indices, symbol2_indices))
            return list(zip(symbol1_indices, symbol2_indices))

    def bind_particle(self, particle):
        symbols = sorted(particle.atoms.get_symbols())
        symbol1 = symbols[0]
        symbol2 = symbols[1]

        symbol1_indices = particle.get_indices_by_symbol(symbol1)
        symbol2_indices = particle.get_indices_by_symbol(symbol2)

        self.n_symbol1_atoms = len(symbol1_indices)
        self.n_symbol2_atoms = len(symbol2_indices)

        self.max_exchanges = min(self.n_symbol1_atoms, self.n_symbol2_atoms)

        atom_features = particle.get_atom_features(self.feature_key)
        for index in symbol1_indices:
            feature = atom_features[index]
            self.symbol1_indices[index] = self.env_energy_differences[self.env_from_feature(feature)]

        for index in symbol2_indices:
            feature = atom_features[index]
            self.symbol2_indices[index] = self.env_energy_differences[self.env_from_feature(feature)]

    def update(self, particle, indices):
        symbols = sorted(particle.atoms.get_symbols())
        symbol1 = symbols[0]

        atom_features = particle.get_atom_features(self.feature_key)
        for index in indices:
            feature = atom_features[index]
            if particle.get_symbol(index) == symbol1:
                self.symbol1_indices[index] = self.env_energy_differences[self.env_from_feature(feature)]
            else:
                self.symbol2_indices[index] = self.env_energy_differences[self.env_from_feature(feature)]

    def reset_index(self):
        self.index = 0

