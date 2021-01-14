import numpy as np

from Core.BaseNanoparticle import BaseNanoparticle

from ase.cluster import Octahedron
from ase import Atoms


class Nanoparticle(BaseNanoparticle):
    def __init__(self):
        BaseNanoparticle.__init__(self)

    def truncated_octahedron(self, height, cutoff, stoichiometry, lattice_constant=3.9, alloy=False):
        octa = Octahedron('Pt', height, cutoff, latticeconstant=lattice_constant, alloy=alloy)
        atoms = Atoms(octa.symbols, octa.positions)

        self.add_atoms(atoms, recompute_neighbor_list=False)
        self.random_ordering(stoichiometry)
        self.construct_neighbor_list()

    def adjust_stoichiometry(self, target_stoichiometry):
        def transform_n_random_atoms(symbol_from, symbol_to, n_atoms):
            symbol_from_atoms = self.get_indices_by_symbol(symbol_from)
            atoms_to_be_transformed = np.random.choice(symbol_from_atoms, n_atoms, replace=False)
            self.transform_atoms(zip(atoms_to_be_transformed, [symbol_to] * n_atoms), a)

        for symbol in self.get_stoichiometry():
            if symbol in target_stoichiometry:
                difference = self.get_stoichiometry()[symbol] - target_stoichiometry[symbol]
                if difference > 0:
                    transform_n_random_atoms(symbol, 'Z', difference)

        for symbol in target_stoichiometry:
            if symbol == 'Z':
                continue
            difference = target_stoichiometry[symbol]
            if symbol in self.get_stoichiometry():
                difference = target_stoichiometry[symbol] - self.get_stoichiometry()[symbol]
            transform_n_random_atoms('Z', symbol, difference)
        return

