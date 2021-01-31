import numpy as np
from ase import Atoms


class CuttingPlane:
    def __init__(self, anchor, normal):
        self.anchor = anchor
        self.normal = normal

    def split_atom_indices(self, atoms):
        dot_product = np.dot((atoms.positions - self.anchor), self.normal)
        indices_in_positive_subspace = dot_product > 0
        indices_in_negative_subspace = dot_product < 0
        return indices_in_positive_subspace, indices_in_negative_subspace


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
        normal = 1 - 2*np.random.random(3)
        normal /= np.linalg.norm(normal)

        anchor_dir = 1 - 2 * np.random.random(3)
        anchor_dir /= np.linalg.norm(anchor_dir)
        anchor = anchor_dir * (self.min_radius + np.random.random() * (self.max_radius - self.min_radius))
        anchor = anchor + self.center

        return CuttingPlane(anchor, normal)
