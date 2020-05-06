import numpy as np

from Core.BaseNanoparticle import BaseNanoparticle
from ase import Atoms


class Nanoparticle(BaseNanoparticle):
    def __init__(self, lattice):
        BaseNanoparticle.__init__(self, lattice)

    def truncated_octahedron(self, height, trunc, stoichiometry):
        bounding_box_anchor = self.lattice.get_anchor_index_of_centered_box(2 * height, 2 * height, 2 * height)
        lower_tip_position = bounding_box_anchor + np.array([height, height, 0])

        if not self.lattice.is_valid_lattice_position(lower_tip_position):
            lower_tip_position[2] = lower_tip_position[2] + 1

        layer_basis_vector1 = np.array([1, 1, 0])
        layer_basis_vector2 = np.array([-1, 1, 0])
        for z_position in range(height):
            if z_position < trunc:
                continue
            layer_width = z_position + 1
            lower_layer_offset = np.array([0, -z_position, z_position])
            upper_layer_offset = np.array([0, -z_position, 2 * height - 2 - z_position])

            lower_layer_start_position = lower_tip_position + lower_layer_offset
            upper_layer_start_position = lower_tip_position + upper_layer_offset
            for x_position in range(layer_width):
                for y_position in range(layer_width):
                    if z_position >= height - trunc:
                        to_be_removed = z_position - (height - trunc) + 1
                        if x_position + y_position < to_be_removed:
                            continue

                        if x_position + y_position > 2*(layer_width - 1) - to_be_removed:
                            continue

                        if y_position > (layer_width - 1 - to_be_removed) + x_position:
                            continue

                        if y_position < 1 - layer_width + to_be_removed + x_position:
                            continue

                    current_position_lower_layer = lower_layer_start_position + x_position * layer_basis_vector1 + y_position * layer_basis_vector2
                    current_position_upper_layer = upper_layer_start_position + x_position * layer_basis_vector1 + y_position * layer_basis_vector2

                    lower_layer_index = self.lattice.get_index_from_lattice_position(current_position_lower_layer)
                    upper_layer_index = self.lattice.get_index_from_lattice_position(current_position_upper_layer)

                    self.atoms.add_atoms([(lower_layer_index, 'Pt'), (upper_layer_index, 'Au')])

        self.construct_neighbor_list()

        transformed_stoichiometry = dict()
        if sum(list(stoichiometry.values())) <= 1:
            n_atoms = self.get_n_atoms()
            print("N atoms: {0}".format(n_atoms))
            for symbol in stoichiometry:
                transformed_stoichiometry[symbol] = int(stoichiometry[symbol]*n_atoms)
            if sum(list(transformed_stoichiometry.values())) != n_atoms:
                difference = n_atoms - sum(list(transformed_stoichiometry.values()))
                transformed_stoichiometry[list(transformed_stoichiometry.keys())[0]] += difference
            self.random_ordering(transformed_stoichiometry)
        else:
            self.random_ordering(stoichiometry)

    def get_ASE_atoms(self, centered=True, exclude_X=True):
        atom_positions = list()
        atomic_symbols = list()
        for lattice_index in self.atoms.get_indices():
            if exclude_X is True:
                if self.atoms.get_symbol(lattice_index) is 'X':
                    continue
            atom_positions.append(self.lattice.get_cartesian_position_from_index(lattice_index))
            atomic_symbols.append(self.atoms.get_symbol(lattice_index))

        atoms = Atoms(positions=atom_positions, symbols=atomic_symbols)
        if centered:
            COM = atoms.get_center_of_mass()
            return Atoms(positions=[position - COM for position in atom_positions], symbols=atomic_symbols)
        else:
            return atoms

    def surface_ordering(self, base_symbol, surface_stoichiometry):
        surface_indices = self.get_atom_indices_from_coordination_number(list(range(12)))
        inner_indices = self.get_inner_atom_indices()
        print(len(surface_indices))
        print(len(inner_indices))

        transformed_stoichiometry = dict()
        if sum(list(surface_stoichiometry.values())) <= 1:
            n_atoms = len(surface_indices)
            print("N surface atoms: {0}".format(n_atoms))
            for symbol in surface_stoichiometry:
                transformed_stoichiometry[symbol] = int(surface_stoichiometry[symbol]*n_atoms)
            if sum(list(transformed_stoichiometry.values())) != n_atoms:
                difference = n_atoms - sum(list(transformed_stoichiometry.values()))
                transformed_stoichiometry[list(transformed_stoichiometry.keys())[0]] += difference

            symbols = [symbol for symbol in transformed_stoichiometry for i in range(transformed_stoichiometry[symbol])]
            np.random.shuffle(surface_indices)
            self.atoms.transform_atoms(zip(surface_indices, symbols))

        else:
            symbols = [symbol for symbol in surface_stoichiometry for i in range(surface_stoichiometry[symbol])]
            np.random.shuffle(surface_indices)
            self.atoms.transform_atoms(zip(surface_indices, symbols))

        self.atoms.transform_atoms(zip(inner_indices, [base_symbol]*len(inner_indices)))
