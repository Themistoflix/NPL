import numpy as np
from collections import defaultdict

class IndexedAtoms:
    def __init__(self):
        self.symbol_by_index = dict()
        self.indices_by_symbol = defaultdict(lambda: [])

    def add_atoms(self, atoms):
        for atom in atoms:
            index = atom[0]
            symbol = atom[1]

            self.symbol_by_index[index] = symbol
            self.indices_by_symbol[symbol].append(index)

    def remove_atoms(self, indices):
        for index in indices:
            symbol = self.symbol_by_index[index]

            self.symbol_by_index.pop(index)
            self.indices_by_symbol[symbol].remove(index)

    def get_atoms(self, indices=None):
        if indices is None:
            indices = self.get_indices()
        symbols = [self.symbol_by_index[index] for index in indices]

        return list(zip(indices, symbols))

    def clear(self):
        self.symbol_by_index.clear()
        self.indices_by_symbol.clear()

    def swap_atoms(self, pairs):
        for pair in pairs:
            index1 = pair[0]
            index2 = pair[1]

            symbol1 = self.symbol_by_index[index1]
            symbol2 = self.symbol_by_index[index2]

            self.symbol_by_index[index1] = symbol2
            self.symbol_by_index[index2] = symbol1

            self.indices_by_symbol[symbol1].remove(index1)
            self.indices_by_symbol[symbol2].append(index1)

            self.indices_by_symbol[symbol2].remove(index2)
            self.indices_by_symbol[symbol1].append(index2)

    def random_ordering(self, stoichiometry):
        new_ordering = list()
        for symbol in stoichiometry:
            for i in range(stoichiometry[symbol]):
                new_ordering.append(symbol)

        np.random.shuffle(new_ordering)

        self.indices_by_symbol.clear()

        for symbol_index, atom_index in enumerate(self.symbol_by_index):
            new_symbol = new_ordering[symbol_index]
            self.symbol_by_index[atom_index] = new_symbol
            if new_symbol in self.indices_by_symbol:
                self.indices_by_symbol[new_symbol].append(atom_index)
            else:
                same_symbol_atoms = list()
                same_symbol_atoms.append(atom_index)
                self.indices_by_symbol[new_symbol] = same_symbol_atoms

    def transform_atoms(self, new_atoms):
        for atom in new_atoms:
            index = atom[0]
            newSymbol = atom[1]
            oldSymbol = self.symbol_by_index[index]

            self.symbol_by_index[index] = newSymbol
            self.indices_by_symbol[oldSymbol].remove(index)
            self.indices_by_symbol[newSymbol].append(index)

    def get_indices(self):
        return list(self.symbol_by_index)

    def get_symbols(self):
        symbols = list()
        for symbol in self.indices_by_symbol:
            if not self.indices_by_symbol[symbol] is []:
                symbols.append(symbol)

        return symbols

    def get_symbol(self, index):
        return self.symbol_by_index[index]

    def get_indices_by_symbol(self, symbol):
        if symbol in self.indices_by_symbol.keys():
            return self.indices_by_symbol[symbol]
        else:
            return []

    def get_n_atoms(self, include_X=True):
        if include_X:
            return len(self.symbol_by_index)
        else:
            n_atoms = 0
            for symbol in self.get_symbols():
                if symbol != 'X':
                    n_atoms += self.get_n_atoms_of_symbol(symbol)
            return n_atoms

    def get_n_atoms_of_symbol(self, symbol):
        if symbol in self.indices_by_symbol.keys():
            return len(self.indices_by_symbol[symbol])
        else:
            return 0

    def get_stoichiometry(self):
        stoichiometry = defaultdict(lambda: 0)
        for symbol in self.indices_by_symbol:
            stoichiometry[symbol] = len(self.indices_by_symbol[symbol])

        return stoichiometry
