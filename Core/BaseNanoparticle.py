import numpy as np
import copy
import pickle

from ase import Atoms

from Core.BoundingBox import BoundingBox
from Core.IndexedAtoms import IndexedAtoms, Atom
from Core.NeighborList import NeighborList


class BaseNanoparticle:
    def __init__(self, lattice):
        self.lattice = lattice
        self.atoms = IndexedAtoms()
        self.neighbor_list = NeighborList()
        self.bounding_box = BoundingBox()

        self.energies = dict()

        self.local_environments = dict()
        self.atom_features = dict()
        self.feature_vectors = dict()

    def get_topological_data(self):
        data = dict()
        data['neighbor_list'] = self.neighbor_list.list
        data['positions'] = self.atoms.get_positions()

        return data

    def get_as_dictionary(self, thin=False):
        data = dict()
        data['energies'] = self.energies
        data['symbols'] = self.atoms.get_symbols()

        if not thin:
            data['atom_features'] = self.atom_features
            data['local_environments'] = self.local_environments
            data['neighbor_list'] = self.neighbor_list.list
            data['feature_vectors'] = self.feature_vectors

            data['positions'] = self.atoms.get_positions()

        return data

    def save(self, filename, thin=False, save_topological_data_extra=False, filename_topo=None):
        data = self.get_as_dictionary(thin)
        pickle.dump(data, open(filename, 'wb'))

        if save_topological_data_extra is True:
            topological_data = self.get_topological_data()
            pickle.dump(topological_data, open(filename_topo, 'wb'))

    def build_from_dictionary(self, dictionary=None, topological_data=None):
        if dictionary is not None:
            symbols = dictionary['symbols']
            n_atoms = len(symbols)
            if 'positions' in dictionary:
                atoms = [Atom(symbols[i], dictionary['positions'][i]) for i in range(n_atoms)]
            else:
                atoms = [Atom(symbols[i], topological_data['positions'][i]) for i in range(n_atoms)]
        else:
            n_atoms = len(topological_data['positions'])
            symbols = ['X']*n_atoms
            atoms = [Atom(symbols[i], topological_data['positions'][i]) for i in range(n_atoms)]
        self.atoms.add_atoms(atoms)

        self.neighbor_list = NeighborList()
        if dictionary is not None:
            if 'neighbor_list' in dictionary:
                self.neighbor_list.list = dictionary['neighbor_list']
        if topological_data is not None:
            if 'neighbor_list' in topological_data:
                self.neighbor_list.list = topological_data['neighbor_list']

        if dictionary is not None:
            self.energies = dictionary['energies']

            if 'feature_vectors' in dictionary:
                self.feature_vectors = dictionary['feature_vectors']
            if 'atom_features' in dictionary:
                self.atom_features = dictionary['atom_features']

            if 'local_environments' in dictionary:
                self.local_environments = dictionary['local_environments']

    def load(self, filename=None, filename_topological_data=None):
        if filename is not None:
            dictionary = pickle.load(open(filename, 'rb'))
        else:
            dictionary = None

        if filename_topological_data is not None:
            topological_data = pickle.load(open(filename_topological_data, 'rb'))
            self.build_from_dictionary(dictionary, topological_data)
        else:
            self.build_from_dictionary(dictionary)

    def load_xyz(self, filename, scale_factor=1.0, construct_neighbor_list=True):
        with open(filename) as file:
            for line in file.readlines():
                s = line.split(' ')
                s = [f for f in s if f != '']
                symbol = s[0]
                position = np.array([float(s[1]), float(s[2]), float(s[3])])*scale_factor
                atom = Atom(symbol, position)
                self.atoms.add_atoms([atom])

        if construct_neighbor_list:
            self.construct_neighbor_list()

    def add_atoms(self, atoms):
        self.atoms.add_atoms(atoms)
        # TODO handle neighborlist
        #indices, _ = zip(*atoms)
        #self.neighbor_list.add_atoms(list(indices))

    def remove_atoms(self, lattice_indices):
        self.atoms.remove_atoms(lattice_indices)

        # TODO handle neighbor list
        #self.neighbor_list.remove_atoms(lattice_indices)

    def transform_atoms(self, new_atoms, new_symbols):
        self.atoms.transform_atoms(new_atoms, new_symbols)

    def get_indices(self):
        return self.atoms.get_indices()

    def get_n_bonds(self):
        return self.neighbor_list.get_n_bonds()

    def get_contributing_symbols(self):
        return self.atoms.get_contributing_symbols()

    def get_symbol(self, index):
        return self.atoms.get_symbol(index)

    def get_indices_by_symbol(self, symbol):
        return self.atoms.get_indices_by_symbol(symbol)

    def random_ordering(self, stoichiometry):
        self.atoms.random_ordering(stoichiometry)

    def construct_neighbor_list(self):
        self.neighbor_list.construct(self.atoms)

    def construct_bounding_box(self):
        self.bounding_box.construct(self.lattice, self.atoms.get_indices())

    def get_atom_indices_from_coordination_number(self, coordination_numbers, symbol=None):
        if symbol is None:
            return list(filter(lambda x: self.get_coordination_number(x) in coordination_numbers, self.atoms.get_indices()))
        else:
            return list(filter(lambda x: self.get_coordination_number(x) in coordination_numbers and self.atoms.get_symbol(x) == symbol, self.atoms.get_indices()))

    def get_coordination_number(self, lattice_index):
        return self.neighbor_list.get_coordination_number(lattice_index)

    def get_atoms(self, atomIndices=None):
        return copy.deepcopy(self.atoms.get_atoms(atomIndices))

    def get_n_atoms(self):
        return self.atoms.get_n_atoms()

    def get_neighbor_list(self):
        return self.neighbor_list

    def get_ASE_atoms(self, centered=True):
        positions = [atom.get_position() for atom in self.atoms.get_atoms()]
        symbols = [atom.get_symbol() for atom in self.atoms.get_atoms()]

        atoms = Atoms(positions=positions, symbols=symbols)
        if centered:
            center_of_mass = atoms.get_center_of_mass()
            positions = [atom.get_position() - center_of_mass for atom in self.atoms.get_atoms()]
            atoms = Atoms(positions=positions, symbols=symbols)

        return atoms

    def get_stoichiometry(self):
        return self.atoms.get_stoichiometry()

    def get_n_atoms_of_symbol(self, symbol):
        return self.atoms.get_n_atoms_of_symbol(symbol)

    def set_energy(self, key, energy):
        self.energies[key] = energy

    def get_energy(self, key):
        return self.energies[key]

    def has_energy(self, key):
        if key in self.energies:
            return True
        return False

    def set_feature_vector(self, key, feature_vector):
        self.feature_vectors[key] = feature_vector

    def get_feature_vector(self, key):
        return self.feature_vectors[key]

    def set_atom_features(self, atom_features, feature_key):
        self.atom_features[feature_key] = atom_features

    def set_atom_feature(self, feature_key, index, atom_feature):
        self.atom_features[feature_key][index] = atom_feature

    def get_atom_features(self, feature_key):
        if feature_key not in self.atom_features:
            self.atom_features[feature_key] = dict()
        return self.atom_features[feature_key]

    def set_local_environment(self, lattice_index, local_environment):
        self.local_environments[lattice_index] = local_environment

    def get_local_environment(self, lattice_index):
        return self.local_environments[lattice_index]

    def set_local_environments(self, local_environments):
        self.local_environments = local_environments

    def get_local_environments(self):
        return self.local_environments

    def is_pure(self):
        first_symbol = True
        for symbol in self.atoms.get_contributing_symbols():
            if self.atoms.get_n_atoms_of_symbol(symbol) > 0:
                if first_symbol:
                    first_symbol = False
                else:
                    return False
        return True
