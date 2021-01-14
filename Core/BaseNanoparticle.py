import pickle

from Core.BoundingBox import BoundingBox
from Core.AtomWrapper import AtomWrapper
from Core.NeighborList import NeighborList

from ase import Atoms
from ase.io import read, write


class BaseNanoparticle:
    def __init__(self):
        self.atoms = AtomWrapper()
        self.neighbor_list = NeighborList()

        # TODO bounding box necessary?
        self.bounding_box = BoundingBox()

        self.energies = dict()

        self.local_environments = dict()
        self.atom_features = dict()
        self.feature_vectors = dict()

    def get_geometrical_data(self):
        # Do not include symbols here so we can reuse the same geometry for several atomic orderings
        data = dict()
        data['neighbor_list'] = self.neighbor_list.list
        data['positions'] = self.atoms.get_positions()

        return data

    def get_as_dictionary(self, fields=None):
        full_particle_dict = {'energies': self.energies,
                              'symbols': list(self.atoms.get_symbols()),
                              'positions': self.atoms.get_positions(),
                              'atom_features': self.atom_features,
                              'local environments': self.local_environments,
                              'neighbor_list': self.neighbor_list.list,
                              'feature_vectors': self.feature_vectors}

        if fields is None:
            return full_particle_dict
        else:
            data = {}
            for field in fields:
                data[field] = full_particle_dict[field]
            return data

    def save(self, filename, fields, filename_geometry=None):
        data = self.get_as_dictionary(fields)
        pickle.dump(data, open(filename, 'wb'))

        if filename_geometry is not None:
            geometrical_data = self.get_geometrical_data()
            pickle.dump(geometrical_data, open(filename_geometry, 'wb'))

    def build_from_dictionary(self, particle_dict=None, geometrical_dict=None):
        if geometrical_dict is None:
            positions = particle_dict['positions']
            symbols = particle_dict['symbols']
        else:
            positions = geometrical_dict['positions']
            if particle_dict is None:
                symbols = ['X']*len(positions)
            else:
                symbols = particle_dict['symbols']

        atoms = Atoms(symbols, positions)
        self.atoms.add_atoms(atoms)

        if particle_dict is None:
            self.neighbor_list.list = geometrical_dict['neighbor_list']
        else:
            if 'neighbor_list' in particle_dict:
                self.neighbor_list.list = particle_dict['neighbor_list']

        if particle_dict is not None:
            if 'neighbor_list' in particle_dict:
                self.neighbor_list.list = particle_dict['neighbor_list']
        if geometrical_dict is not None:
            if 'neighbor_list' in geometrical_dict:
                self.neighbor_list.list = geometrical_dict['neighbor_list']

        if particle_dict is not None:
            self.energies = particle_dict['energies']

            if 'feature_vectors' in particle_dict:
                self.feature_vectors = particle_dict['feature_vectors']
            if 'atom_features' in particle_dict:
                self.atom_features = particle_dict['atom_features']

            if 'local_environments' in particle_dict:
                self.local_environments = particle_dict['local_environments']

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

    def load_xyz(self, filename, construct_neighbor_list=True):
        atoms = read(filename)
        self.atoms.add_atoms(atoms)

        if construct_neighbor_list:
            self.construct_neighbor_list()

    def save_xyz(self, filename):
        atoms = self.atoms.get_ase_atoms()
        write(filename, atoms)

    def add_atoms(self, atoms, recompute_neighbor_list=True):
        self.atoms.add_atoms(atoms)

        if recompute_neighbor_list:
            self.construct_neighbor_list()

    def remove_atoms(self, atom_indices, recompute_neighbor_list=True):
        self.atoms.remove_atoms(atom_indices)

        if recompute_neighbor_list:
            self.construct_neighbor_list()

    def swap_symbols(self, index_pairs):
        self.atoms.swap_symbols(index_pairs)

    def transform_atoms(self, atom_indices, new_symbols):
        self.atoms.transform_atoms(atom_indices, new_symbols)

    def random_ordering(self, stoichiometry):
        # account for stoichiometries given as proportions instead of absolute numbers
        if sum(stoichiometry.values()) == 1:
            n_atoms = self.atoms.get_n_atoms()
            transformed_stoichiometry = dict()
            for symbol in sorted(stoichiometry):
                transformed_stoichiometry[symbol] = int(n_atoms*stoichiometry[symbol])

            # adjust for round-off error
            if sum(transformed_stoichiometry.values()) != n_atoms:
                diff = n_atoms - sum(transformed_stoichiometry.values())
                transformed_stoichiometry[sorted(stoichiometry)[0]] += diff

            print(transformed_stoichiometry)
            self.atoms.random_ordering(transformed_stoichiometry)
        else:
            self.atoms.random_ordering(stoichiometry)

    def get_indices(self):
        # TODO necessary?
        return self.atoms.get_indices()

    def get_n_bonds(self):
        return self.neighbor_list.get_n_bonds()

    def get_all_symbols(self):
        return self.atoms.get_all_symbols()

    def get_symbol(self, index):
        return self.atoms.get_symbol(index)

    def get_symbols(self, indices=None):
        return self.atoms.get_symbols(indices)

    def get_indices_by_symbol(self, symbol):
        return self.atoms.get_indices_by_symbol(symbol)

    def construct_neighbor_list(self):
        self.neighbor_list.construct(self.get_ase_atoms())

    def get_atom_indices_from_coordination_number(self, coordination_numbers, symbol=None):
        if symbol is None:
            return list(filter(lambda x: self.get_coordination_number(x) in coordination_numbers, self.atoms.get_indices()))
        else:
            return list(filter(lambda x: self.get_coordination_number(x) in coordination_numbers
                                         and self.atoms.get_symbol(x) == symbol, self.atoms.get_indices()))

    def get_coordination_number(self, atom_idx):
        return self.neighbor_list.get_coordination_number(atom_idx)

    def get_n_atoms(self):
        return self.atoms.get_n_atoms()

    def get_neighbor_list(self):
        return self.neighbor_list

    def get_ase_atoms(self, indices=None):
        return self.atoms.get_ase_atoms(indices)

    def get_stoichiometry(self):
        return self.atoms.get_stoichiometry()

    def get_n_atoms_of_symbol(self, symbol):
        return self.atoms.get_n_atoms_of_symbol(symbol)

    def set_energy(self, energy_key, energy):
        self.energies[energy_key] = energy

    def get_energy(self, energy_key):
        return self.energies[energy_key]

    def has_energy(self, energy_key):
        if energy_key in self.energies:
            return True
        return False

    def set_feature_vector(self, feature_key, feature_vector):
        self.feature_vectors[feature_key] = feature_vector

    def get_feature_vector(self, feature_key):
        return self.feature_vectors[feature_key]

    def set_atom_features(self, atom_features, feature_key):
        self.atom_features[feature_key] = atom_features

    def set_atom_feature(self, feature_key, index, atom_feature):
        self.atom_features[feature_key][index] = atom_feature

    def get_atom_features(self, feature_key):
        if feature_key not in self.atom_features:
            self.atom_features[feature_key] = dict()
        return self.atom_features[feature_key]

    def set_local_environment(self, atom_idx, local_environment):
        self.local_environments[atom_idx] = local_environment

    def get_local_environment(self, atom_idx):
        return self.local_environments[atom_idx]

    def set_local_environments(self, local_environments):
        self.local_environments = local_environments

    def get_local_environments(self):
        return self.local_environments

    def is_pure(self):
        if len(self.atoms.get_stoichiometry()) == 1:
            return True
        return False
