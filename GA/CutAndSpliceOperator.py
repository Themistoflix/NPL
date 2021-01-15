from Core.Nanoparticle import Nanoparticle
from Core.CuttingPlaneUtilities import SphericalCuttingPlaneGenerator


class CutAndSpliceOperator:
    def __init__(self, max_radius, recompute_neighbor_list=False):
        self.cutting_plane_generator = SphericalCuttingPlaneGenerator(max_radius, 0.0, 0.0)
        self.recompute_neighbor_list = recompute_neighbor_list

    def cut_and_splice(self, particle1, particle2, fixed_stoichiometry=True):
        self.cutting_plane_generator.set_center(particle1.get_ase_atoms().get_center_of_mass())

        while True:
            cutting_plane = self.cutting_plane_generator.generate_new_cutting_plane()
            atom_indices_in_positive_subspace, _ = cutting_plane.split_atoms(particle1.get_ase_atoms())
            _, atom_indices_in_negative_subspace = cutting_plane.split_atoms(particle2.get_ase_atoms())

            if len(atom_indices_in_negative_subspace) > 0 and len(atom_indices_in_positive_subspace) > 0:
                break

        new_particle = Nanoparticle()
        new_particle.add_atoms(atom_indices_in_negative_subspace)
        new_particle.add_atoms(atom_indices_in_positive_subspace)

        if fixed_stoichiometry is True:
            target_stoichiometry = particle1.get_stoichiometry()
            new_particle.adjust_stoichiometry(target_stoichiometry)

        if self.recompute_neighbor_list:
            new_particle.construct_neighbor_list()

        return new_particle
