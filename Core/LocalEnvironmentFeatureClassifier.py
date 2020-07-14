import numpy as np
from sklearn.cluster import KMeans
import copy


class LocalEnvironmentFeatureClassifier:
    def __init__(self, local_environment_calculator):
        self.local_environment_calculator = local_environment_calculator
        self.feature_key = None

    def compute_atom_features(self, particle, recompute_local_environments=False):
        if recompute_local_environments:
            self.local_environment_calculator.compute_local_environments(particle)

        for index in particle.get_indices():
            self.compute_atom_feature(particle, index, recompute_local_environments)

    def compute_feature_vector(self, particle, recompute_atom_features=True, recompute_local_environments=False):
        if recompute_atom_features:
            self.compute_atom_features(particle, recompute_local_environments)

        n_features = self.compute_n_features(particle)
        feature_vector = np.zeros(n_features)
        atom_features = particle.get_atom_features(self.feature_key)
        for index in particle.get_indices():
            feature_vector[atom_features[index]] += 1

        particle.set_feature_vector(self.feature_key, feature_vector)

    def compute_atom_feature(self, particle, lattice_index, recompute_local_environment=False):
        feature = self.predict_atom_feature(particle, lattice_index, recompute_local_environment)
        atom_features = particle.get_atom_features(self.feature_key)
        atom_features[lattice_index] = feature

    def get_feature_key(self):
        return self.feature_key

    def set_feature_key(self, feature_key):
        self.feature_key = feature_key

    def compute_n_features(self, particle):
        raise NotImplementedError

    def predict_atom_feature(self, particle, lattice_index, recompute_local_environment=False):
        raise NotImplementedError


class KMeansClassifier(LocalEnvironmentFeatureClassifier):
    def __init__(self, n_cluster, local_environment_calculator, feature_key):
        LocalEnvironmentFeatureClassifier.__init__(self, local_environment_calculator)
        self.kMeans = None
        self.n_cluster = n_cluster
        self.feature_key = feature_key

    def compute_n_features(self, particle):
        n_elements = len(particle.get_contributing_symbols())
        n_features = self.n_cluster * n_elements
        return n_features

    def predict_atom_feature(self, particle, lattice_index, recompute_local_environment=False):
        symbol = particle.get_symbol(lattice_index)
        symbols = sorted(particle.get_contributing_symbols())
        symbol_index = symbols.index(symbol)

        offset = symbol_index*self.n_cluster
        if recompute_local_environment:
            environment = self.kMeans.predict([self.local_environment_calculator.predict_local_environment(particle, lattice_index)])[0]
        else:
            environment = self.kMeans.predict([particle.get_local_environment(lattice_index)])[0]
        return offset + environment

    def kmeans_clustering(self, training_set):
        local_environments = list()
        for particle in training_set:
            local_environments = local_environments + list(particle.get_local_environments().values())

        print("Starting kMeans")
        self.kMeans = KMeans(n_clusters=self.n_cluster, random_state=0).fit(local_environments)


class TopologicalEnvironmentClassifier(LocalEnvironmentFeatureClassifier):
    def __init__(self, local_environment_calculator):
        LocalEnvironmentFeatureClassifier.__init__(self, local_environment_calculator)
        self.feature_key = 'TEC'

    def compute_n_features(self, particle):
        return 76

    def predict_atom_feature(self, particle, lattice_index, recompute_local_environment=False):
        symbol = particle.get_symbol(lattice_index)
        symbols = sorted(particle.get_contributing_symbols())
        symbol_index = symbols.index(symbol)

        element_offset = symbol_index*38

        if recompute_local_environment:
            self.local_environment_calculator.compute_local_environment(particle, lattice_index)

        environment = particle.get_local_environment(lattice_index)
        coordination_number = np.sum(environment)
        coordination_number_offsets = {6: 0, 7: 7, 9: 15, 12: 25}

        atom_feature = element_offset + coordination_number_offsets[coordination_number] + environment[0]

        return atom_feature


class TopologicalEnvironmentClassifier2(LocalEnvironmentFeatureClassifier):
    def __init__(self, local_environment_calculator, symbols):
        LocalEnvironmentFeatureClassifier.__init__(self, local_environment_calculator)
        symbols_copy = copy.deepcopy(symbols)
        symbols_copy.sort()
        self.symbols = symbols_copy

        self.feature_key = 'TEC'

    def compute_n_features(self, particle):
        return 94

    def predict_atom_feature(self, particle, lattice_index, recompute_local_environment=False):
        symbol = particle.get_symbol(lattice_index)
        symbol_index = self.symbols.index(symbol)

        element_offset = symbol_index*47

        if recompute_local_environment:
            self.local_environment_calculator.compute_local_environment(particle, lattice_index)

        environment = particle.get_local_environment(lattice_index)
        coordination_number = len(particle.neighbor_list[lattice_index])
        coordination_number_offsets = {6: 0, 7: 7, 8: 15, 9: 24, 12: 34}

        atom_feature = element_offset + coordination_number_offsets[coordination_number] + environment[0]

        return atom_feature


class TopologicalEnvironmentClassifier3(LocalEnvironmentFeatureClassifier):
    def __init__(self, local_environment_calculator):
        LocalEnvironmentFeatureClassifier.__init__(self, local_environment_calculator)
        self.feature_key = 'TEC'

        self.coordination_number_offsets = dict()
        self.coordination_number_offsets[0] = 0
#        for n_symbol_a_atoms in range(13):
 #           self.coordination_number_offsets[n_symbol_a_atoms] = sum([13 - i for i in range(n_symbol_a_atoms)])

        for n_symbol_a_atoms in range(13):
            self.coordination_number_offsets[n_symbol_a_atoms] = n_symbol_a_atoms
        self.n_envs = 13

    def compute_n_features(self, particle):
        return self.n_envs*2

    def predict_atom_feature(self, particle, lattice_index, recompute_local_environment=False):
        symbol = particle.get_symbol(lattice_index)
        symbols = sorted(particle.get_contributing_symbols())
        symbol_index = symbols.index(symbol)

        element_offset = symbol_index*self.n_envs

        if recompute_local_environment:
            self.local_environment_calculator.compute_local_environment(particle, lattice_index)

        environment = particle.get_local_environment(lattice_index)
        coordination_number = environment[0]

        atom_feature = element_offset + self.coordination_number_offsets[coordination_number]

        return atom_feature


