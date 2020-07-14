import numpy as np
from sortedcontainers import SortedKeyList


class GuidedExchangeOperator:
    def __init__(self, local_energies, total_energies, feature_key):
        self.local_energies = local_energies
        self.n_envs = int(len(local_energies)/2)
        self.env_energy_differences = [total_energies[i] - total_energies[i + self.n_envs] for i in range(self.n_envs)]

        self.feature_key = feature_key

        self.symbol1_exchange_energies = dict()
        self.symbol2_exchange_energies = dict()

        self.symbol1_indices = SortedKeyList(key=lambda x: self.symbol1_exchange_energies[x])
        self.symbol2_indices = SortedKeyList(key=lambda x: self.symbol2_exchange_energies[x])

        self.index = 0
        self.n_symbol1_atoms = 0
        self.n_symbol2_atoms = 0

        self.max_exchanges = 0

    def env_from_feature(self, x):
        return x % self.n_envs


    def guided_exchange(self, particle):
        symbol1_index = self.symbol1_indices[self.index % self.n_symbol1_atoms]
        symbol1_energy = self.symbol1_exchange_energies[symbol1_index]

        symbol2_index = self.symbol2_indices[self.index % self.n_symbol2_atoms]
        symbol2_energy = self.symbol2_exchange_energies[symbol2_index]

        expected_energy_gain = symbol1_energy + symbol2_energy
        self.index += 1

        particle.atoms.swap_atoms([(symbol1_index, symbol2_index)])
        return [(symbol1_index, symbol2_index)]

    def basin_hop_step(self, particle):
        expected_energy_gain = -1
        self.index = -1
        while expected_energy_gain <= 0 and self.index < min(self.n_symbol1_atoms, self.n_symbol2_atoms):
            self.index += 1
            symbol1_index = self.symbol1_indices[self.index % self.n_symbol1_atoms]
            symbol1_energy = self.symbol1_exchange_energies[symbol1_index]

            symbol2_index = self.symbol2_indices[self.index % self.n_symbol2_atoms]
            symbol2_energy = self.symbol2_exchange_energies[symbol2_index]

            expected_energy_gain = symbol1_energy + symbol2_energy
            if expected_energy_gain > 0:
                self.index = 0
                particle.atoms.swap_atoms([(symbol1_index, symbol2_index)])
                return [(symbol1_index, symbol2_index)]

        symbol1_index = self.symbol1_indices[self.index % self.n_symbol1_atoms]
        symbol2_index = self.symbol2_indices[self.index % self.n_symbol2_atoms]
        self.index = 0
        particle.atoms.swap_atoms([(symbol1_index, symbol2_index)])
        return [(symbol1_index, symbol2_index)]

    def stochastic_guided_exchange(self, particle):
        n_exchanges = min(np.random.geometric(p=self.p_geometric, size=1)[0], self.max_exchanges)

        sorted_symbol1_exchange_energies = np.array([self.symbol1_exchange_energies[index] for index in self.symbol1_indices])
        weights = np.exp(-self.beta*sorted_symbol1_exchange_energies)
        weights = weights/np.sum(weights)
        symbol1_indices = np.random.choice(self.symbol1_indices, n_exchanges, replace=False, p=weights)

        sorted_symbol2_exchange_energies = np.array([self.symbol2_exchange_energies[index] for index in self.symbol2_indices])
        weights = np.exp(-self.beta * sorted_symbol2_exchange_energies)
        weights = weights / np.sum(weights)
        symbol2_indices = np.random.choice(self.symbol2_indices, n_exchanges, replace=False, p=weights)

        #expected_energy_gain = symbol1_energy + symbol2_energy

        '''if expected_energy_gain < 0:
            particle.atoms.swap_atoms([(symbol1_index, symbol2_index)])
            return [(symbol1_index, symbol2_index)]
        else:
            n_exchanges = min(np.random.geometric(p=self.p_geometric, size=1)[0], self.max_exchanges)
            symbol1_indices = np.random.choice(self.symbol1_indices, n_exchanges, replace=False)
            symbol2_indices = np.random.choice(self.symbol2_indices, n_exchanges, replace=False)

            particle.atoms.swap_atoms(zip(symbol1_indices, symbol2_indices))
            return list(zip(symbol1_indices, symbol2_indices))'''

        particle.atoms.swap_atoms(zip(symbol1_indices, symbol2_indices))
        return list(zip(symbol1_indices, symbol2_indices))

    def bind_particle(self, particle):
        symbols = sorted(particle.atoms.get_contributing_symbols())
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
            self.symbol1_exchange_energies[index] = -self.env_energy_differences[self.env_from_feature(feature)]
            self.symbol1_indices.add(index)

        for index in symbol2_indices:
            feature = atom_features[index]
            self.symbol2_exchange_energies[index] = self.env_energy_differences[self.env_from_feature(feature)]
            self.symbol2_indices.add(index)

    def update(self, particle, indices, exchange_indices):
        symbols = sorted(particle.atoms.get_contributing_symbols())
        symbol1 = symbols[0]

        atom_features = particle.get_atom_features(self.feature_key)
        for index in indices:
            if index in exchange_indices:
                if particle.get_symbol(index) == symbol1:
                    self.symbol2_indices.remove(index)
                else:
                    self.symbol1_indices.remove(index)
            else:
                if particle.get_symbol(index) == symbol1:
                    self.symbol1_indices.remove(index)
                else:
                    self.symbol2_indices.remove(index)

        for index in indices:
            feature = atom_features[index]
            new_exchange_energy = self.env_energy_differences[self.env_from_feature(feature)]
            if index in exchange_indices:
                if particle.get_symbol(index) == symbol1:
                    self.symbol1_exchange_energies[index] = -new_exchange_energy
                    del self.symbol2_exchange_energies[index]
                else:
                    self.symbol2_exchange_energies[index] = new_exchange_energy
                    del self.symbol1_exchange_energies[index]
            else:
                if particle.get_symbol(index) == symbol1:
                    self.symbol1_exchange_energies[index] = -new_exchange_energy
                else:
                    self.symbol2_exchange_energies[index] = new_exchange_energy

        for index in indices:
            if particle.get_symbol(index) == symbol1:
                self.symbol1_indices.add(index)
            else:
                self.symbol2_indices.add(index)

    def reset_index(self):
        self.index = 0


class RandomExchangeOperator:
    def __init__(self, p_geometric):
        self.symbol1 = None
        self.symbol2 = None

        self.n_symbol1_atoms = 0
        self.n_symbol2_atoms = 0

        self.max_exchanges = 0
        self.p_geometric = p_geometric

    def bind_particle(self, particle):
        symbols = sorted(particle.atoms.get_contributing_symbols())
        self.symbol1 = symbols[0]
        self.symbol2 = symbols[1]

        symbol1_indices = particle.get_indices_by_symbol(self.symbol1)
        symbol2_indices = particle.get_indices_by_symbol(self.symbol2)

        self.n_symbol1_atoms = len(symbol1_indices)
        self.n_symbol2_atoms = len(symbol2_indices)

        self.max_exchanges = min(self.n_symbol1_atoms, self.n_symbol2_atoms)

    def random_exchange(self, particle):
        n_exchanges = min(np.random.geometric(p=self.p_geometric, size=1)[0], self.max_exchanges)
        symbol1_indices = np.random.choice(particle.get_indices_by_symbol(self.symbol1), n_exchanges, replace=False)
        symbol2_indices = np.random.choice(particle.get_indices_by_symbol(self.symbol2), n_exchanges, replace=False)

        particle.atoms.swap_atoms(zip(symbol1_indices, symbol2_indices))
        return list(zip(symbol1_indices, symbol2_indices))

