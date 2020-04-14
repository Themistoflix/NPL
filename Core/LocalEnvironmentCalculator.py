import numpy as np
import copy
from scipy.special import sph_harm


class LocalEnvironmentCalculator:
    def __init__(self):
        pass

    def compute_local_environments(self, particle):
        for index in particle.get_indices():
            self.compute_local_environment(particle, index)

    def compute_local_environment(self, particle, lattice_index):
        local_env = self.predict_local_environment(particle, lattice_index)
        particle.set_local_environment(lattice_index, local_env)

    def predict_local_environment(self, particle, lattice_index):
        raise NotImplementedError


class SOAPCalculator(LocalEnvironmentCalculator):
    def __init__(self, l_max):
        LocalEnvironmentCalculator.__init__(self)
        self.l_max = l_max

    def predict_local_environment(self, particle, lattice_index):
        def map_onto_unit_sphere(cartesian_coordinates):
            # note the use of the scipy.special.sph_harm notation for phi and theta (which is the opposite of wikipedias)
            def angular_from_cartesian_coordinates(cartesian_coordinates):
                x = cartesian_coordinates[0]
                y = cartesian_coordinates[1]
                z = cartesian_coordinates[2]

                hxy = np.hypot(x, y)
                r = np.hypot(hxy, z)
                el = np.arctan2(z, hxy)
                az = np.arctan2(y, x)
                return np.abs(az + np.pi), np.abs(el + np.pi / 2.0)

            return list(map(lambda x: angular_from_cartesian_coordinates(x), cartesian_coordinates))

        def spherical_harmonics_expansion():
            """
            This functions takes the environment atoms surrounding a reference atom
            in an fcc lattice and returns the spherical harmonic coefficients of the expansion
            """

            neighbors = particle.get_atomic_neighbors(lattice_index)
            symbols = np.array([particle.get_symbol(index) for index in neighbors])
            cartesian_coordinates = [particle.lattice.get_cartesian_position_from_index(index) for index in neighbors]

            center_atom_position = particle.lattice.get_cartesian_position_from_index(lattice_index)
            cartesian_coordinates = list(map(lambda x: x - center_atom_position, cartesian_coordinates))

            angular_coordinates = map_onto_unit_sphere(cartesian_coordinates)

            # The density of each species is expanded separately
            expansion_coefficients = []
            n_neighbors = len(symbols)
            for symbol in sorted(particle.get_symbols()):
                symbol_density = np.zeros(n_neighbors)
                symbol_density[np.where(symbols == symbol)] += 1
                c_lms_symbol = []
                for l in range(self.l_max + 1):
                    for m in range(-l, l + 1):
                        c_lm = 0.0
                        for i, point in enumerate(angular_coordinates):
                            c_lm += symbol_density[i] * np.conj(sph_harm(m, l, point[0], point[1]))
                        c_lms_symbol.append(c_lm)
                expansion_coefficients.append(c_lms_symbol)
            return expansion_coefficients

        sh_expansion_coefficients = spherical_harmonics_expansion()
        bond_parameters = []
        for symbol_index_1, symbol_1 in enumerate(sorted(particle.get_symbols())):
            for symbol_index_2, symbol_2 in enumerate(sorted(particle.get_symbols())):
                n_neighbors_with_symbol_1 = max(1, len(
                    list(filter(lambda x: particle.get_symbol(x) == symbol_1, particle.get_atomic_neighbors(lattice_index)))))
                n_neighbors_with_symbol_2 = max(1, len(
                    list(filter(lambda x: particle.get_symbol(x) == symbol_2,
                                particle.get_atomic_neighbors(lattice_index)))))
                q_ls_symbol = []
                i = 0
                for l in range(self.l_max + 1):
                    q_l = 0
                    for m in range(-l, l + 1):
                        q_l += 1.0 / (n_neighbors_with_symbol_1 * n_neighbors_with_symbol_2) * np.conj(sh_expansion_coefficients[symbol_index_1][i]) * sh_expansion_coefficients[symbol_index_2][i]
                        i += 1
                    q_ls_symbol.append(np.sqrt((np.sqrt(4.0 * np.pi) / (2. * l + 1.)) * q_l))
                bond_parameters.append(q_ls_symbol)

        bond_parameters = np.array(bond_parameters)
        bond_parameters = bond_parameters.real
        # return the bond parameters as one big feature vector
        bond_parameters = np.reshape(bond_parameters, bond_parameters.size)

        return bond_parameters


class NeighborCountingEnvironmentCalculator(LocalEnvironmentCalculator):
    def __init__(self, symbols):
        LocalEnvironmentCalculator.__init__(self)
        symbols_copy = copy.deepcopy(symbols)
        symbols_copy.sort()
        self.symbol_a = symbols_copy[0]
        self.symbol_b = symbols_copy[1]

    def predict_local_environment(self, particle, lattice_index):
        n_a_atoms = 0
        n_b_atoms = 0

        neighbor_list = particle.neighbor_list
        neighbors = neighbor_list[lattice_index]
        for neighbor in neighbors:
            if particle.get_symbol(neighbor) == self.symbol_a:
                n_a_atoms += 1
            else:
                n_b_atoms += 1

        return np.array([n_a_atoms, n_b_atoms])



