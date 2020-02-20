import numpy as np

from Core.BaseNanoparticle import BaseNanoparticle


class Nanoparticle(BaseNanoparticle):
    def __init__(self, lattice):
        BaseNanoparticle.__init__(self, lattice)
        self.fitness = 0

    def truncated_octahedron(self, height, symbols, n_atoms_same_symbol):
        bounding_box_anchor = self.lattice.get_anchor_index_of_centered_box(2 * height, 2 * height, 2 * height)
        lower_tip_position = bounding_box_anchor + np.array([height, height, 0])

        if not self.lattice.is_valid_lattice_position(lower_tip_position):
            lower_tip_position[2] = lower_tip_position[2] + 1

        layer_basis_vector1 = np.array([1, 1, 0])
        layer_basis_vector2 = np.array([-1, 1, 0])
        for z_position in range(height):
            layer_width = z_position + 1
            lower_layer_offset = np.array([0, -z_position, z_position])
            upper_layer_offset = np.array([0, -z_position, 2 * height - 2 - z_position])

            lower_layer_start_position = lower_tip_position + lower_layer_offset
            upper_layer_start_position = lower_tip_position + upper_layer_offset
            for width in range(layer_width):
                for length in range(layer_width):
                    current_position_lower_layer = lower_layer_start_position + width * layer_basis_vector1 + length * layer_basis_vector2
                    current_position_upper_layer = upper_layer_start_position + width * layer_basis_vector1 + length * layer_basis_vector2

                    lower_layer_index = self.lattice.get_index_from_lattice_position(current_position_lower_layer)
                    upper_layer_index = self.lattice.get_index_from_lattice_position(current_position_upper_layer)

                    self.atoms.add_atoms([(lower_layer_index, 'X'), (upper_layer_index, 'X')])

        self.construct_neighbor_list()
        corners = self.get_atom_indices_from_coordination_number([4])

        self.remove_atoms(corners)
        self.random_ordering(symbols, n_atoms_same_symbol)

