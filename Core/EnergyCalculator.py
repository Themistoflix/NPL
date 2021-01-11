from ase.optimize import BFGS
from asap3 import EMT

import numpy as np
import copy
import sklearn.gaussian_process as gp
from sklearn.linear_model import BayesianRidge


class EnergyCalculator:
    def __init__(self):
        self.energy_key = None
        pass

    def compute_energy(self, particle):
        raise NotImplementedError

    def get_energy_key(self):
        return copy.deepcopy(self.energy_key)

    def set_energy_key(self, energy_key):
        self.energy_key = energy_key


class EMTCalculator(EnergyCalculator):
    def __init__(self, fmax=0.01, steps=20):
        EnergyCalculator.__init__(self)
        self.fmax = fmax
        self.steps = steps
        self.energy_key = 'EMT'

    def compute_energy(self, particle, return_optimized_atoms=False):
        cell_width = 1e3
        cell_height = 1e3
        cell_length = 1e3

        # TODO check if this modifies the particle
        # in that case add relax atoms kwarg
        atoms = particle.get_ase_atoms()
        atoms.set_cell(np.array([[cell_width, 0, 0], [0, cell_length, 0], [0, 0, cell_height]]))
        atoms.set_calculator(EMT())
        dyn = BFGS(atoms)
        dyn.run(fmax=self.fmax, steps=self.steps)

        energy = atoms.get_potential_energy()
        particle.set_energy(self.energy_key, energy)

        if return_optimized_atoms:
            return atoms


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

    def fit(self, training_set, energy_key):
        feature_vectors = [p.get_feature_vector(self.feature_key) for p in training_set]
        energies = [p.get_energy(energy_key) for p in training_set]

        self.GPR = gp.GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=20, alpha=self.alpha, normalize_y=self.normalize_y)
        self.GPR.fit(feature_vectors, energies)

    def compute_energy(self, particle):
        energy = self.GPR.predict([particle.get_feature_vector(self.feature_key)])[0]
        particle.set_energy(self.energy_key, energy)


class MixingEnergyCalculator(EnergyCalculator):
    # TODO decouple from EMT, couple with any other energy calculator
    def __init__(self, mixing_parameters=None, fmax=0.05, steps=20, recompute_emt_energy=False):
        EnergyCalculator.__init__(self)

        if mixing_parameters is None:
            self.mixing_parameters = dict()
        else:
            self.mixing_parameters = mixing_parameters

        self.emt_calculator = EMTCalculator(fmax=fmax, steps=steps)
        self.recompute_emt_energy = recompute_emt_energy
        self.energy_key = 'Mixing Energy'

    def compute_mixing_parameters(self, particle, symbols):
        n_atoms = particle.atoms.get_n_atoms()
        for symbol in symbols:
            particle.random_ordering({symbol: n_atoms})
            self.emt_calculator.compute_energy(particle)
            self.mixing_parameters[symbol] = particle.get_energy('EMT')

    def compute_energy(self, particle):
        if self.recompute_emt_energy:
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

    def fit(self, training_set, energy_key):
        feature_vectors = [p.get_feature_vector(self.feature_key) for p in training_set]
        energies = [p.get_energy(energy_key) for p in training_set]

        self.ridge.fit(feature_vectors, energies)

    def get_coefficients(self):
        return self.ridge.coef_

    def set_coefficients(self, new_coefficients):
        self.ridge.coef_ = new_coefficients

    def set_feature_key(self, feature_key):
        self.feature_key = feature_key

    def compute_energy(self, particle):
        brr_energy = np.dot(np.transpose(self.ridge.coef_), particle.get_feature_vector(self.feature_key))
        particle.set_energy(self.energy_key, brr_energy)

# TODO move to relevant file -> Basin Hopping
def compute_coefficients_for_linear_topological_model(global_topological_coefficients, symbols, n_atoms):
    coordination_numbers = list(range(13))
    symbols_copy = copy.deepcopy(symbols)
    symbols_copy.sort()
    symbol_a = symbols_copy[0]
    print("Coef symbol_a: {}".format(symbol_a))

    E_aa_bond = global_topological_coefficients[0]/n_atoms
    E_bb_bond = global_topological_coefficients[1]/n_atoms
    E_ab_bond = global_topological_coefficients[2]/n_atoms

    coefficients = []
    total_energies = []
    for symbol in symbols_copy:
        for cn_number in coordination_numbers:
            for n_symbol_a_atoms in range(cn_number + 1):
                E = 0
                E_tot = 0
                if symbol == symbol_a:
                    E += (global_topological_coefficients[3]*0.1) # careful...
                    E += (n_symbol_a_atoms*E_aa_bond/2)
                    E += ((cn_number - n_symbol_a_atoms)*E_ab_bond/2)
                    E += (global_topological_coefficients[4 + cn_number])

                    E_tot = E
                    E_tot += n_symbol_a_atoms*E_aa_bond/2
                    E_tot += (cn_number - n_symbol_a_atoms)*E_ab_bond/2
                else:
                    E += (n_symbol_a_atoms*E_ab_bond/2)
                    E += ((cn_number - n_symbol_a_atoms)*E_bb_bond/2)

                    E_tot = E
                    E_tot += n_symbol_a_atoms*E_ab_bond/2
                    E_tot += (cn_number - n_symbol_a_atoms)*E_bb_bond/2

                coefficients.append(E)
                total_energies.append(E_tot)

    coefficients = np.array(coefficients)

    return coefficients, total_energies


def compute_coefficients_for_shape_optimization(global_topological_coefficients, symbols, n_atoms):
    coordination_numbers = list(range(13))
    symbols_copy = copy.deepcopy(symbols)
    symbols_copy.sort()
    symbol_a = symbols_copy[0]

    E_aa_bond = global_topological_coefficients[0]

    coordination_energies_a = dict()
    for index, cn in enumerate(coordination_numbers):
        coordination_energies_a[cn] = global_topological_coefficients[index + 1]

    coefficients = []
    total_energies = []
    for symbol in symbols_copy:
        for n_symbol_a_atoms in coordination_numbers:
            E = 0
            E_tot = 0
            if symbol == symbol_a:
                E += (n_symbol_a_atoms*E_aa_bond/2)
                E += (coordination_energies_a[n_symbol_a_atoms])

                E_tot += n_symbol_a_atoms*E_aa_bond/2
                coefficients.append(E)
                total_energies.append(E_tot)
    coefficients += [0]*len(coordination_numbers)
    total_energies += [0]*len(coordination_numbers)

    coefficients = np.array(coefficients)

    return coefficients
