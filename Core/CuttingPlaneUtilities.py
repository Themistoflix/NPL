import numpy as np
from ase import Atoms


class CuttingPlane:
    def __init__(self, anchor, normal):
        self.anchor = anchor
        self.normal = normal

    def split_atoms(self, atoms):
        atoms_in_positive_subspace = Atoms()
        atoms_in_negative_subspace = Atoms()

        for atom in atoms.copy():
            position = atom.position
            if np.dot((position - self.anchor), self.normal) >= 0.0:
                atoms_in_positive_subspace += atom
            else:
                atoms_in_negative_subspace += atom
        return atoms_in_positive_subspace, atoms_in_negative_subspace


class SphericalCuttingPlaneGenerator:
    def __init__(self, max_radius, min_radius=0.0, center=0.0):
        self.center = center
        self.min_radius = min_radius
        self.max_radius = max_radius

    def set_center(self, center):
        self.center = center

    def set_max_radius(self, max_radius):
        self.max_radius = max_radius

    def generate_new_cutting_plane(self):
        normal = np.array([np.random.random() for _ in range(3)])
        normal = normal / np.linalg.norm(normal)
        anchor = normal * (self.min_radius + np.random.normal(0, 1) * (self.max_radius - self.min_radius))
        anchor = anchor + self.center

        return CuttingPlane(anchor, normal)
