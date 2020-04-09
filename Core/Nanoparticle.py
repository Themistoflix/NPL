import numpy as np
from collections import defaultdict

from Core.BaseNanoparticle import BaseNanoparticle


class Nanoparticle(BaseNanoparticle):
    def __init__(self, lattice):
        BaseNanoparticle.__init__(self, lattice)
        self.exchange_energies = defaultdict(lambda: dict())

    def compute_exchange_energy(self, index_a, index_b, local_energies, local_feature_classifier):
        self.atoms.swap_atoms([(index_a, index_b)])

        feature_key = local_feature_classifier.get_feature_key()
        neighbors_a = self.neighbor_list[index_a]
        neigbhors_b = self.neighbor_list[index_b]

        exchange_energy = 0
        old_features = self.get_atom_features(feature_key)
        for atom_index in neighbors_a | neigbhors_b | {index_a} | {index_b}:
            old_feature = old_features[atom_index]
            new_feature = local_feature_classifier.predict_atom_feature(self, atom_index, True)

            exchange_energy += local_energies[new_feature] - local_energies[old_feature]

        self.atoms.swap_atoms([(index_a, index_b)])

        return exchange_energy

    def compute_exchange_energies(self, local_energies, local_feature_classifier, indices):
        symbol_a = sorted(self.atoms.get_symbols())[0]
        symbol_b = sorted(self.atoms.get_symbols())[1]

        indices_symbol_a = list(filter(lambda x: self.get_symbol(x) == symbol_a, indices))
        indices_symbol_b = list(filter(lambda x: self.get_symbol(x) == symbol_b, indices))

        for index_a in indices_symbol_a:
            for index_b in indices_symbol_b:
                self.exchange_energies[index_a][index_b] = self.compute_exchange_energy(index_a, index_b, local_energies, local_feature_classifier)

    def get_exchange_energies_as_list(self):
        exchange_energies = []
        for index_a in self.exchange_energies.keys():
            for index_b in self.exchange_energies[index_a].keys():
                exchange_energies.append((index_a, index_b, self.exchange_energies[index_a][index_b]))

        return exchange_energies

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

                    self.atoms.add_atoms([(lower_layer_index, 'X'), (upper_layer_index, 'X')])

        self.construct_neighbor_list()

        if sum(list(stoichiometry.values())) <= 1:
            n_atoms = self.get_n_atoms()
            for symbol in stoichiometry:
                stoichiometry[symbol] = int(stoichiometry[symbol]*n_atoms)
            if sum(list(stoichiometry.values())) != n_atoms:
                difference = n_atoms - sum(list(stoichiometry.values()))
                stoichiometry[list(stoichiometry.keys())[0]] += difference

        self.random_ordering(stoichiometry)
