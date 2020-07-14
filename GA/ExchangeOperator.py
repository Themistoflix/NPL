import numpy as np
import copy


class ExchangeOperator:
    def __init__(self, p):
        self.p = p
        self.index = 0

    def random_exchange(self, particle, n_exchanges=None):
        new_particle = copy.deepcopy(particle)
        if new_particle.is_pure():
            print("Pure particle! No permutation possible")
            return new_particle

        symbols = new_particle.atoms.get_contributing_symbols()
        symbol1 = symbols[0]
        symbol2 = symbols[1]

        max_exchanges = min(len(new_particle.atoms.get_indices_by_symbol(symbol1)), len(new_particle.atoms.get_indices_by_symbol(symbol2)))
        if n_exchanges is None:
            n_exchanges = min(self.draw_from_geometric_distribution(self.p), max_exchanges)

        symbol1_indices = np.random.choice(new_particle.atoms.get_indices_by_symbol(symbol1), n_exchanges, replace=False)
        symbol2_indices = np.random.choice(new_particle.atoms.get_indices_by_symbol(symbol2), n_exchanges, replace=False)

        new_particle.atoms.swap_atoms(zip(symbol1_indices, symbol2_indices))

        return new_particle

    def draw_from_geometric_distribution(self, p):
        return np.random.geometric(p=p, size=1)[0]





