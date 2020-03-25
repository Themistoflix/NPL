import numpy as np


class GlobalFeatureClassifier:
    def __init__(self):
        self.feature_key = None

    def get_feature_key(self):
        return self.feature_key

    def compute_feature_vector(self, particle):
        raise NotImplementedError


class SimpleFeatureClassifier(GlobalFeatureClassifier):
    def __init__(self, symbol_a, symbol_b):
        GlobalFeatureClassifier.__init__(self)
        self.symbol_a = symbol_a
        self.symbol_b = symbol_b

        self.feature_key = 'SFC'
        return

    def compute_feature_vector(self, particle):
        n_aa_bonds, n_bb_bonds, n_ab_bonds = self.compute_respective_bond_counts(particle)
        n_atoms = particle.atoms.get_n_atoms()

        M = particle.get_stoichiometry()[self.symbol_a] * 0.1
        particle.set_feature_vector(self.feature_key, np.array([n_aa_bonds / n_atoms, n_bb_bonds / n_atoms, n_ab_bonds / n_atoms, M]))

    def compute_respective_bond_counts(self, particle):
        n_aa_bonds = 0
        n_ab_bonds = 0
        n_bb_bonds = 0

        for lattice_index_with_symbol_a in particle.atoms.get_indices_by_symbol(self.symbol_a):
            neighbor_list = particle.neighbor_list[lattice_index_with_symbol_a]
            for neighbor in neighbor_list:
                symbol_neighbor = particle.atoms.get_symbol(neighbor)

                if self.symbol_a != symbol_neighbor:
                    n_ab_bonds += 1
                else:
                    n_aa_bonds += 1

        for lattice_index_with_symbol_b in particle.atoms.get_indices_by_symbol(self.symbol_b):
            neighbor_list = particle.neighbor_list[lattice_index_with_symbol_b]
            for neighbor in neighbor_list:
                symbol_neighbor = particle.atoms.get_symbol(neighbor)

                if self.symbol_b == symbol_neighbor:
                    n_bb_bonds += 1

        return n_aa_bonds, n_bb_bonds, n_ab_bonds


class TopologicalFeatureClassifier(SimpleFeatureClassifier):
    def __init__(self, symbol_a, symbol_b):
        SimpleFeatureClassifier.__init__(self, symbol_a, symbol_b)
        self.symbol_a = symbol_a
        self.symbol_b = symbol_b

        self.feature_key = 'TFC'

    def compute_feature_vector(self, particle):
        n_atoms = particle.get_n_atoms()
        n_aa_bonds, n_bb_bonds, n_ab_bonds = self.compute_respective_bond_counts(particle)
        n_corners_a = len(particle.get_atom_indices_from_coordination_number([6], symbol=self.symbol_a))
        n_edge = len(particle.get_atom_indices_from_coordination_number([7], symbol=self.symbol_a))
        n_terrace_a = len(particle.get_atom_indices_from_coordination_number([9], symbol=self.symbol_a))

        M = particle.get_stoichiometry()[self.symbol_a] * 0.1

        feature_vector = np.array([n_aa_bonds/n_atoms, n_bb_bonds/n_atoms, n_ab_bonds/n_atoms, M, n_corners_a, n_edge, n_terrace_a])
        particle.set_feature_vector(self.feature_key, feature_vector)
