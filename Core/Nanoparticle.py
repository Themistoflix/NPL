import numpy as np

from Core.BaseNanoparticle import BaseNanoparticle
from Core.AtomWrapper import Atom
from Core import Profiler


class Nanoparticle(BaseNanoparticle):
    def __init__(self, lattice):
        BaseNanoparticle.__init__(self)

    def truncated_octahedron(self, height, trunc, stoichiometry, scale_factor):
        bounding_box_anchor = [0, 0, 0]
        lower_tip_position = bounding_box_anchor + np.array([height, height, 0])

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

                    current_position_lower_layer = scale_factor*current_position_lower_layer
                    current_position_upper_layer = scale_factor*current_position_upper_layer

                    lower_atom = Atom('Pt', current_position_lower_layer)
                    upper_atom = Atom('Au', current_position_upper_layer)
                    if z_position == height - 1:
                        self.atoms.add_atoms([lower_atom])
                    else:
                        self.atoms.add_atoms([upper_atom, lower_atom])

        self.construct_neighbor_list()

        transformed_stoichiometry = dict()
        if sum(list(stoichiometry.values())) <= 1:
            n_atoms = self.get_n_atoms()
            for symbol in stoichiometry:
                transformed_stoichiometry[symbol] = int(stoichiometry[symbol]*n_atoms)
            if sum(list(transformed_stoichiometry.values())) != n_atoms:
                difference = n_atoms - sum(list(transformed_stoichiometry.values()))
                transformed_stoichiometry[list(transformed_stoichiometry.keys())[0]] += difference
            self.random_ordering(transformed_stoichiometry)
        else:
            self.random_ordering(stoichiometry)

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

