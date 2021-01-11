import numpy as np

from ase import Atoms


class AtomWrapper:
    def __init__(self):
        self.atoms = Atoms()

        return

    def add_atoms(self, new_atoms):
        self.atoms.extend(new_atoms)

    def remove_atoms(self, indices):
        del self.atoms[indices]

    def get_ase_atoms(self, indices=None):
        if indices is None:
            return self.atoms
        return self.atoms[indices]

    def swap_symbol(self, index_pairs):
        for idx1, idx2 in index_pairs:
            self.atoms[idx1].symbol, self.atoms[idx2].symbol = self.atoms[idx2].symbol, self.atoms[idx1].symbol

    def random_ordering(self, new_stoichiometry):
        new_symbols = []
        for symbol in new_stoichiometry:
            new_symbols += [symbol]*new_stoichiometry[symbol]
        np.random.shuffle(new_symbols)
        self.atoms.symbols = new_symbols

    def transform_atoms(self, atom_indices, new_symbols):
        for idx, symbol in zip(atom_indices, new_symbols):
            self.atoms[idx].symbol = symbol

    def get_indices(self):
        return np.arange(0, len(self.atoms))

    def get_all_symbols(self):
        # return list of symbols which occur at least once in the particle
        return self.atoms.symbols.species()

    def get_symbol(self, atom_idx):
        return self.atoms[atom_idx].symbol

    def get_symbols(self, indices=None):
        if indices is None:
            return self.atoms.symbols
        return self.atoms[indices].symbols

    def get_indices_by_symbol(self, symbol):
        if symbol in self.atoms.symbols.indices():
            return self.atoms.symbols.indices()[symbol]
        else:
            return None

    def get_n_atoms(self):
        return len(self.atoms)

    def get_n_atoms_of_symbol(self, symbol):
        return len(self.get_indices_by_symbol(symbol))

    def get_stoichiometry(self):
        stoichiometry = self.atoms.symbols.indices()
        for symbol in stoichiometry:
            stoichiometry[symbol] = len(stoichiometry[symbol])
        return stoichiometry

    def get_positions(self, indices=None):
        if indices is None:
            return self.atoms.positions
        return self.atoms[indices].positions


