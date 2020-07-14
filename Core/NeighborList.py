#from ase.neighborlist import NeighborList as ASENeighborList
from ase.neighborlist import natural_cutoffs
from ase.neighborlist import build_neighbor_list
from ase.neighborlist import NewPrimitiveNeighborList
from ase import Atoms
import numpy as np


class NeighborList:
    def __init__(self):
        self.list = dict()

    def __getitem__(self, item):
        return self.list[item]

    def __setitem__(self, key, value):
        self.list[key] = value

    def construct(self, indexed_atoms):
        sorted_atom_indices = sorted([index for index in indexed_atoms.atoms_by_index])

        positions = []
        symbols = []
        for atom_index in sorted_atom_indices:
            atom = indexed_atoms.atoms_by_index[atom_index]
            positions.append(atom.get_position())
            symbols.append(atom.get_symbol())

        atoms = Atoms(positions=positions, symbols=symbols)
        neighbor_list = build_neighbor_list(atoms, cutoffs=natural_cutoffs(atoms), bothways=True, self_interaction=False)


        for i in range(len(sorted_atom_indices)):
            neighbors, _ = neighbor_list.get_neighbors(i)
            self.list[sorted_atom_indices[i]] = set(neighbors)

    def get_coordination_number(self, atom_index):
        return len(self.list[atom_index])

    def get_n_bonds(self):
        n_bonds = sum([len(l) for l in list(self.list.values())])
        return n_bonds/2
