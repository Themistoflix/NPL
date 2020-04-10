import numpy as np
import copy
import pickle

from ase import Atoms

from Core.BoundingBox import BoundingBox
from Core.CuttingPlaneUtilities import CuttingPlane
from Core.IndexedAtoms import IndexedAtoms
from Core.NeighborList import NeighborList
from Core.FCCLattice import FCCLattice


class BaseNanoparticle:
    def __init__(self, lattice):
        self.lattice = lattice
        self.atoms = IndexedAtoms()
        self.neighbor_list = NeighborList(lattice)
        self.bounding_box = BoundingBox()

        self.energies = dict()

        self.local_environments = dict()
        self.atom_features = dict()
        self.feature_vectors = dict()

    def from_particle_data(self, atoms, neighbor_list=None):
        self.atoms = atoms
        if neighbor_list is None:
            self.construct_neighbor_list()
        else:
            self.neighbor_list = neighbor_list()

        self.construct_bounding_box()

    def get_as_dictionary(self):
        data = dict()
        lattice_data = dict()
        lattice_data['width'] = self.lattice.width
        lattice_data['length'] = self.lattice.length
        lattice_data['height'] = self.lattice.height
        lattice_data['lattice_constant'] = self.lattice.lattice_constant

        data['lattice'] = lattice_data
        data['energies'] = self.energies

        data['atom_features'] = self.atom_features
        data['local_environments'] = self.local_environments

        atom_indices = self.atoms.get_indices()
        corresponding_symbols = [self.atoms.get_symbol(index) for index in atom_indices]

        data['atoms'] = {'indices': atom_indices, 'symbols': corresponding_symbols}
        data['neighbor_list'] = self.neighbor_list.list

        return data

    def save(self, filename):
        data = self.get_as_dictionary()
        pickle.dump(data, open(filename, 'wb'))

    def build_from_dictionary(self, dictionary):
        lattice_data = dictionary['lattice']
        self.lattice = FCCLattice(lattice_data['width'], lattice_data['length'], lattice_data['height'], lattice_data['lattice_constant'])
        self.neighbor_list = NeighborList(self.lattice)
        self.neighbor_list.list = dictionary['neighbor_list']

        self.energies = dictionary['energies']
        self.atoms.add_atoms(list(zip(dictionary['atoms']['indices'], dictionary['atoms']['symbols'])))

        self.feature_vectors = dictionary['feature_vectors']
        self.atom_features = dictionary['atom_features']

        self.local_environments = dictionary['local_environments']

    def load(self, filename):
        dictionary = pickle.load(open(filename, 'rb'))
        self.build_from_dictionary(dictionary)

    def add_atoms(self, atoms):
        self.atoms.add_atoms(atoms)
        indices, _ = zip(*atoms)
        self.neighbor_list.add_atoms(list(indices))

    def remove_atoms(self, lattice_indices):
        self.atoms.remove_atoms(lattice_indices)
        self.neighbor_list.remove_atoms(lattice_indices)

    def transform_atoms(self, new_atoms):
        self.atoms.transform_atoms(new_atoms)

    def get_indices(self):
        return self.atoms.get_indices()

    def get_symbols(self):
        return self.atoms.get_symbols()

    def get_symbol(self, index):
        return self.atoms.get_symbol(index)

    def get_indices_by_symbol(self, symbol):
        return self.atoms.get_indices_by_symbol(symbol)

    def random_ordering(self, stoichiometry):
        self.atoms.random_ordering(stoichiometry)

    def rectangular_prism(self, w, l, h, symbol='X'):
        anchor_point = self.lattice.get_anchor_index_of_centered_box(w, l, h)
        for x in range(w):
            for y in range(l):
                for z in range(h):
                    cur_position = anchor_point + np.array([x, y, z])

                    if self.lattice.is_valid_lattice_position(cur_position):
                        lattice_index = self.lattice.get_index_from_lattice_position(cur_position)
                        self.atoms.add_atoms([(lattice_index, symbol)])
        self.construct_neighbor_list()

    def convex_shape(self, stoichiometry, w, l, h, cutting_plane_generator):
        self.rectangular_prism(w, l, h)
        self.construct_bounding_box()
        indices_current_atoms = set(self.atoms.get_indices())

        final_n_atoms = sum(list(stoichiometry.values()))
        MAX_CUTTING_ATTEMPTS = 50
        cur_cutting_attempt = 0
        cutting_plane_generator.set_center(self.bounding_box.get_center())

        while len(indices_current_atoms) > final_n_atoms and cur_cutting_attempt < MAX_CUTTING_ATTEMPTS:
            # create cut plane
            cutting_plane = cutting_plane_generator.generate_new_cutting_plane()

            # count atoms to be removed, if new Count >= final Number remove
            atoms_to_be_removed, atoms_to_be_kept = cutting_plane.split_atom_indices(self.lattice, indices_current_atoms)
            if len(atoms_to_be_removed) != 0.0 and len(indices_current_atoms) - len(atoms_to_be_removed) >= final_n_atoms:
                indices_current_atoms = indices_current_atoms.difference(atoms_to_be_removed)
                cur_cutting_attempt = 0
            else:
                cur_cutting_attempt = cur_cutting_attempt + 1

        if cur_cutting_attempt == MAX_CUTTING_ATTEMPTS:
            # place cutting plane parallel to one of the axes and at the anchor point
            cutting_plane = cutting_plane_generator.create_axis_parallel_cutting_plane(self.bounding_box.position)

            # shift till too many atoms would get removed
            n_atoms_yet_to_be_removed = len(indices_current_atoms) - final_n_atoms
            atoms_to_be_removed = set()
            while len(atoms_to_be_removed) < n_atoms_yet_to_be_removed:
                cutting_plane = CuttingPlane(cutting_plane.anchor + cutting_plane.normal * self.lattice.lattice_constant, cutting_plane.normal)
                atoms_to_be_kept, atoms_to_be_removed = cutting_plane.split_atom_indices(self.lattice, indices_current_atoms)

            # remove atoms till the final number is reached "from the ground up"

            def sort_by_position(atom):
                return self.lattice.get_lattice_position_from_index(atom)[0]

            atoms_to_be_removed = list(atoms_to_be_removed)
            atoms_to_be_removed.sort(key=sort_by_position)
            atoms_to_be_removed = atoms_to_be_removed[:n_atoms_yet_to_be_removed]

            atoms_to_be_removed = set(atoms_to_be_removed)
            indices_current_atoms = indices_current_atoms.difference(atoms_to_be_removed)

        # redistribute the different elements randomly
        self.atoms.clear()
        self.atoms.add_atoms(zip(indices_current_atoms, ['X'] * len(indices_current_atoms)))
        self.atoms.random_ordering(stoichiometry)

        self.construct_neighbor_list()

    def optimize_coordination_numbers(self, steps=15):
        for step in range(steps):
            outer_atoms = self.get_atom_indices_from_coordination_number(range(9))
            outer_atoms.sort(key=lambda x: self.get_coordination_number(x))

            start_index = outer_atoms[0]
            symbol = self.atoms.get_symbol(start_index)

            surface_vacancies = list(self.get_surface_vacancies())
            surface_vacancies.sort(key=lambda x: self.get_n_atomic_neighbors(x), reverse=True)

            end_index = surface_vacancies[0]
            if self.get_coordination_number(start_index) < self.get_n_atomic_neighbors(end_index):
                self.remove_atoms([start_index])
                self.add_atoms([(end_index, symbol)])
            else:
                break

        self.construct_neighbor_list()

    def construct_neighbor_list(self):
        self.neighbor_list.construct(self.atoms.get_indices())

    def construct_bounding_box(self):
        self.bounding_box.construct(self.lattice, self.atoms.get_indices())

    def get_corner_atom_indices(self, symbol=None):
        corner_coordination_numbers = [1, 2, 3, 4]
        return self.get_atom_indices_from_coordination_number(corner_coordination_numbers, symbol)

    def get_edge_indices(self, symbol=None):
        edge_coordination_numbers = [5, 6, 7]
        return self.get_atom_indices_from_coordination_number(edge_coordination_numbers, symbol)

    def get_surface_atom_indices(self, symbol=None):
        surface_coordination_numbers = [8, 9]
        return self.get_atom_indices_from_coordination_number(surface_coordination_numbers, symbol)

    def get_terrace_atom_indices(self, symbol=None):
        terrace_coordination_numbers = [10, 11]
        return self.get_atom_indices_from_coordination_number(terrace_coordination_numbers, symbol)

    def get_inner_atom_indices(self, symbol=None):
        innerCoordinationNumbers = [12]
        return self.get_atom_indices_from_coordination_number(innerCoordinationNumbers, symbol)

    def get_n_homo_and_heterogenous_bonds(self):
        n_heteroatomic_bonds = 0
        n_homogenous_bonds = 0

        symbol = self.atoms.get_symbols()[0]

        for lattice_index_with_symbol in self.atoms.get_indices_by_symbol(symbol):
            neighbor_list = self.neighbor_list[lattice_index_with_symbol]
            for neighbor in neighbor_list:
                symbol_of_neighbor = self.atoms.get_symbol(neighbor)

                if symbol != symbol_of_neighbor:
                    n_heteroatomic_bonds = n_heteroatomic_bonds + 1
                else:
                    n_homogenous_bonds += 1

        return n_homogenous_bonds, n_heteroatomic_bonds

    def get_atom_indices_from_coordination_number(self, coordination_numbers, symbol=None):
        if symbol is None:
            return list(filter(lambda x: self.get_coordination_number(x) in coordination_numbers, self.atoms.get_indices()))
        else:
            return list(filter(lambda x: self.get_coordination_number(x) in coordination_numbers and self.atoms.get_symbol(x) == symbol, self.atoms.get_indices()))

    def get_coordination_number(self, lattice_index):
        return self.neighbor_list.get_coordination_number(lattice_index)

    def get_atoms(self, atomIndices=None):
        return copy.deepcopy(self.atoms.get_atoms(atomIndices))

    def get_neighbor_list(self):
        return self.neighbor_list

    def get_ASE_atoms(self, centered=True):
        atom_positions = list()
        atomic_symbols = list()
        for lattice_index in self.atoms.get_indices():
            atom_positions.append(self.lattice.get_cartesian_position_from_index(lattice_index))
            atomic_symbols.append(self.atoms.get_symbol(lattice_index))

        atoms = Atoms(positions=atom_positions, symbols=atomic_symbols)
        if centered:
            COM = atoms.get_center_of_mass()
            return Atoms(positions=[position - COM for position in atom_positions], symbols=atomic_symbols)
        else:
            return atoms

    def get_stoichiometry(self):
        return self.atoms.get_stoichiometry()

    def get_n_atoms(self):
        return self.atoms.get_n_atoms()

    def get_n_atoms_of_symbol(self, symbol):
        return self.atoms.get_n_atoms_of_symbol(symbol)

    def get_surface_vacancies(self):
        not_fully_coordinated_atoms = self.get_atom_indices_from_coordination_number(range(self.lattice.MAX_NEIGHBORS))
        surface_vacancies = set()

        for atom in not_fully_coordinated_atoms:
            neighbor_vacancies = self.lattice.get_nearest_neighbors(atom).difference(self.neighbor_list[atom])
            surface_vacancies = surface_vacancies.union(neighbor_vacancies)
        return surface_vacancies

    def get_atomic_neighbors(self, index):
        neighbors = list()
        nearest_neighbors = self.lattice.get_nearest_neighbors(index)
        for lattice_index in nearest_neighbors:
            if lattice_index in self.atoms.get_indices():
                neighbors.append(lattice_index)

        return neighbors

    def get_n_atomic_neighbors(self, index):
        return len(self.get_atomic_neighbors(index))

    def set_energy(self, key, energy):
        self.energies[key] = energy

    def get_energy(self, key):
        return self.energies[key]

    def set_feature_vector(self, key, feature_vector):
        self.feature_vectors[key] = feature_vector

    def get_feature_vector(self, key):
        return self.feature_vectors[key]

    def set_atom_features(self, atom_features, feature_key):
        self.atom_features[feature_key] = atom_features

    def set_atom_feature(self, feature_key, index, atom_feature):
        self.atom_features[feature_key][index] = atom_feature

    def get_atom_features(self, feature_key):
        if not feature_key in self.atom_features:
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
        for symbol in self.atoms.get_symbols():
            if self.atoms.get_n_atoms_of_symbol(symbol) > 0:
                if first_symbol:
                    first_symbol = False
                else:
                    return False
        return True



