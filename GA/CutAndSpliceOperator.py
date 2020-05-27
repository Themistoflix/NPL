from Core import Nanoparticle as NP
from Core import IndexedAtoms as IA
from Core import CuttingPlaneUtilities


class CutAndSpliceOperator:
    def __init__(self, min_radius, max_radius, center=0.0):
        self.cutting_plane_generator = CuttingPlaneUtilities.SphericalCuttingPlaneGenerator(min_radius, max_radius, center)

    def cut_and_splice(self, particle1, particle2, fixed_stoichiometry=True):
        self.cutting_plane_generator.set_center(particle1.bounding_box.get_center())
        self.cutting_plane_generator.maxRadius = particle1.bounding_box.length/2
        common_lattice = particle1.lattice

        # make sure that we actually cut
        while True:
            cutting_plane = self.cutting_plane_generator.generate_new_cutting_plane()
            atom_indices_in_positive_subspace, _ = cutting_plane.split_atom_indices(common_lattice, particle1.atoms.get_indices())
            _, atom_indices_in_negative_subspace = cutting_plane.split_atom_indices(common_lattice, particle2.atoms.get_indices())

            if len(atom_indices_in_negative_subspace) > 0 and len(atom_indices_in_positive_subspace) > 0:
                break

        new_atom_data = IA.IndexedAtoms()
        new_atom_data.add_atoms(particle1.get_atoms(atom_indices_in_positive_subspace))
        new_atom_data.add_atoms(particle2.get_atoms(atom_indices_in_negative_subspace))

        new_particle = NP.Nanoparticle(common_lattice)
        new_particle.from_particle_data(new_atom_data, particle1.neighbor_list, particle1.bounding_box)

        if fixed_stoichiometry is True:
            target_stoichiometry = particle1.get_stoichiometry()
            new_particle.adjust_stoichiometry(target_stoichiometry)

        return new_particle


