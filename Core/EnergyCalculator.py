from ase.optimize import BFGS
from asap3 import EMT

import numpy as np
import sklearn.gaussian_process as gp
from sklearn.linear_model import BayesianRidge


class EnergyCalculator:
    def __init__(self):
        self.energy_key = None
        pass

    def compute_energy(self, particle):
        raise NotImplementedError

    def get_energy_key(self):
        return self.energy_key


class EMTCalculator(EnergyCalculator):
    def __init__(self, fmax=0.01, steps=20):
        EnergyCalculator.__init__(self)
        self.fmax = fmax
        self.steps = steps
        self.energy_key = 'EMT'

    def compute_energy(self, particle, feature_key=None):
        cell_width = particle.lattice.width * particle.lattice.lattice_constant
        cell_length = particle.lattice.length * particle.lattice.lattice_constant
        cell_height = particle.lattice.height * particle.lattice.lattice_constant

        atoms = particle.get_ASE_atoms()
        atoms.set_cell(np.array([[cell_width, 0, 0], [0, cell_length, 0], [0, 0, cell_height]]))
        atoms.set_calculator(EMT())
        dyn = BFGS(atoms)
        dyn.run(fmax=self.fmax, steps=self.steps)

        energy = atoms.get_potential_energy()
        particle.set_energy(self.energy_key, energy)


class GPRCalculator(EnergyCalculator):
    def __init__(self, feature_key, kernel=None, alpha=0.01, normalize_y=True):
        EnergyCalculator.__init__(self)
        if kernel is None:
            self.kernel = gp.kernels.ConstantKernel(1., (1e-1, 1e3)) * gp.kernels.RBF(1., (1e-3, 1e3))
        else:
            self.kernel = kernel

        self.alpha = alpha
        self.normalize_y = normalize_y
        self.GPR = None
        self.energy_key = 'GPR'
        self.feature_key = feature_key

    def train(self, training_set, energy_key):
        feature_vectors = [p.get_feature_vector(self.feature_key) for p in training_set]
        energies = [p.get_energy(energy_key) for p in training_set]

        self.GPR = gp.GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=20, alpha=self.alpha, normalize_y=self.normalize_y)
        self.GPR.fit(feature_vectors, energies)

    def compute_energy(self, particle):
        energy = self.GPR.predict([particle.get_feature_vector(self.feature_key)])[0]
        particle.set_energy(self.energy_key, energy)


class MixingEnergyCalculator(EnergyCalculator):
    def __init__(self, mixing_parameters=None, fmax=0.05, steps=20):
        EnergyCalculator.__init__(self)

        if mixing_parameters is None:
            self.mixing_parameters = dict()
        else:
            self.mixing_parameters = mixing_parameters

        self.emt_calculator = EMTCalculator(fmax=fmax, steps=steps)
        self.energy_key = 'Mixing Energy'

    def compute_mixing_parameters(self, particle, symbols):
        n_atoms = particle.atoms.get_n_atoms()
        for symbol in symbols:
            particle.random_ordering([symbol], [n_atoms])
            self.emt_calculator.compute_energy(particle)
            self.mixing_parameters[symbol] = particle.get_energy('EMT')

    def compute_energy(self, particle):
        self.emt_calculator.compute_energy(particle)
        mixing_energy = particle.get_energy('EMT')
        n_atoms = particle.atoms.get_n_atoms()

        for symbol in particle.get_stoichiometry():
            mixing_energy -= self.mixing_parameters[symbol] * particle.get_stoichiometry()[symbol] / n_atoms

        particle.set_energy(self.energy_key, mixing_energy)


class BayesianRRCalculator(EnergyCalculator):
    def __init__(self, feature_key):
        EnergyCalculator.__init__(self)

        self.ridge = BayesianRidge(fit_intercept=False)
        self.energy_key = 'BRR'
        self.feature_key = feature_key

    def train(self, training_set, energy_key):
        feature_vectors = [p.get_feature_vector(self.feature_key) for p in training_set]
        energies = [p.get_energy(energy_key) for p in training_set]

        self.ridge.fit(feature_vectors, energies)

    def get_BRR_coefficients(self):
        return self.ridge.coef_

    def compute_energy(self, particle):
        brr_energy = np.dot(np.transpose(self.ridge.coef_), particle.get_feature_vector(self.feature_key))
        particle.set_energy(self.energy_key, brr_energy)