import numpy as np
from sklearn.cluster import KMeans


class LocalEnvironmentFeatureClassifier:
    def __init__(self, local_environment_calculator):
        self.local_environment_calculator = local_environment_calculator
        self.feature_key = None

    def compute_features_as_index_list(self, particle, recompute_local_environments=False):
        if recompute_local_environments:
            self.local_environment_calculator.compute_local_environments(particle)

        n_features = self.compute_n_features(particle)

        features_as_index_lists = list()  # need empty list in case of recalculation
        for i in range(n_features):
            l = list()
            features_as_index_lists.append(l)

        for index in particle.get_indices():
            feature = self.predict_atom_feature(particle, index)
            features_as_index_lists[feature].append(index)

        particle.set_features_as_index_lists(self.feature_key, features_as_index_lists)

    def compute_feature_vector(self, particle, recompute_local_environments=False):
        self.compute_features_as_index_list(particle, recompute_local_environments)

        n_features = self.compute_n_features(particle)
        feature_vector = np.array([len(particle.get_features_as_index_lists(self.feature_key)[feature]) for feature in range(n_features)])

        particle.set_feature_vector(self.feature_key, feature_vector)

    def get_feature_key(self):
        return self.feature_key

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
        n_elements = len(particle.get_symbols())
        n_features = self.n_cluster * n_elements
        return n_features

    def predict_atom_feature(self, particle, lattice_index, recompute_local_environment=False):
        symbol = particle.get_symbol(lattice_index)
        symbols = sorted(particle.get_symbols())
        symbol_index = symbols.index(symbol)

        offset = symbol_index*self.n_cluster
        if recompute_local_environment:
            environment = self.kMeans.predict([self.local_environment_calculator.compute_local_environment(particle, lattice_index)])[0]
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
        symbols = sorted(particle.get_symbols())
        symbol_index = symbols.index(symbol)

        element_offset = symbol_index*38

        if recompute_local_environment:
            env = self.local_environment_calculator.compute_local_environment(particle, lattice_index)
            particle.set_local_environment(lattice_index, env)

        environment = particle.get_local_environment(lattice_index)
        coordination_number = np.sum(environment)
        coordination_number_offsets = {6: 0, 7: 7, 9: 15, 12: 25}

        atom_feature = element_offset + coordination_number_offsets[coordination_number] + environment[0]

        return atom_feature



