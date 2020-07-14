import numpy as np
from collections import defaultdict


class Atom:
    def __init__(self, symbol, position):
        self.symbol = symbol
        self.position = position

    def get_symbol(self):
        return self.symbol

    def get_position(self):
        return self.position

    def set_symbol(self, symbol):
        self.symbol = symbol

    def set_position(self, position):
        self.position = position


class IndexedAtoms:
    def __init__(self):
        self.atoms_by_index = dict()
        self.indices_by_symbol = defaultdict(lambda: [])

        self.max_index = 0

    def get_next_index(self):
        next_index = self.max_index
        self.max_index += 1

        return next_index

    def add_atoms(self, atoms):
        for atom in atoms:
            index = self.get_next_index()
            symbol = atom.get_symbol()

            self.atoms_by_index[index] = atom
            self.indices_by_symbol[symbol].append(index)

    def remove_atoms(self, indices):
        for index in indices:
            symbol = self.atoms_by_index[index].get_symbol()

            self.atoms_by_index.pop(index)
            self.indices_by_symbol[symbol].remove(index)

    def get_atoms(self, indices=None):
        if indices is None:
            indices = self.get_indices()
        atoms = [self.atoms_by_index[index] for index in indices]

        return atoms

    def clear(self):
        self.atoms_by_index.clear()
        self.indices_by_symbol.clear()
        self.max_index = 0

    def swap_atoms(self, pairs):
        for pair in pairs:
            index1 = pair[0]
            index2 = pair[1]

            atom1 = self.atoms_by_index[index1]
            atom2 = self.atoms_by_index[index2]

            symbol1 = atom1.get_symbol()
            symbol2 = atom2.get_symbol()

            atom1.set_symbol(symbol2)
            atom2.set_symbol(symbol1)

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
        for symbol_index, atom_index in enumerate(self.atoms_by_index):
            new_symbol = new_ordering[symbol_index]
            self.atoms_by_index[atom_index].set_symbol(new_symbol)
            self.indices_by_symbol[new_symbol].append(atom_index)

    def transform_atoms(self, atom_indices, new_symbols):
        for atom_index, new_symbol in zip(atom_indices, new_symbols):
            old_symbol = self.atoms_by_index[atom_index].get_symbol()

            self.atoms_by_index[atom_index].set_symbol(new_symbol)
            self.indices_by_symbol[old_symbol].remove(atom_index)
            self.indices_by_symbol[new_symbol].append(atom_index)

    def get_indices(self):
        return sorted(list(self.atoms_by_index))

    def get_contributing_symbols(self):
        symbols = list()
        for symbol in self.indices_by_symbol:
            if not self.indices_by_symbol[symbol] is []:
                symbols.append(symbol)

        return symbols

    def get_symbols(self, indices=None):
        if indices is None:
            indices = self.get_indices()
        return [self.atoms_by_index[index].get_symbol() for index in indices]

    def get_symbol(self, index):
        return self.atoms_by_index[index].get_symbol()

    def get_indices_by_symbol(self, symbol):
        if symbol in self.indices_by_symbol.keys():
            return self.indices_by_symbol[symbol]
        else:
            return []

    def get_n_atoms(self):
        return len(self.atoms_by_index)

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

    def get_positions(self, indices=None):
        if indices is None:
            indices = self.get_indices()
        return [self.atoms_by_index[index].get_position() for index in indices]
