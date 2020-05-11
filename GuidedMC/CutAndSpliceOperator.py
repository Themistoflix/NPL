from Core import Nanoparticle as NP
import copy


class CutAndSpliceOperator:
    def __init__(self, cutting_plane_generator, fill_symbol):
        self.cutting_plane_generator = cutting_plane_generator
        self.fill_symbol = fill_symbol

    def cut_and_splice(self, particle1, particle2):
        self.cutting_plane_generator.set_center(particle1.bounding_box.get_center())
        common_lattice = particle1.lattice

        x_indices = copy.deepcopy(particle1.get_indices_by_symbol('X'))
        particle1.remove_atoms(x_indices)

        x_indices = copy.deepcopy(particle2.get_indices_by_symbol('X'))
        particle2.remove_atoms(x_indices)

        # make sure that we actually cut
        while True:
            cutting_plane = self.cutting_plane_generator.generate_new_cutting_plane()
            atom_indices_in_positive_subspace, _ = cutting_plane.split_atom_indices(common_lattice, particle1.atoms.get_indices())
            _, atom_indices_in_negative_subspace = cutting_plane.split_atom_indices(common_lattice, particle2.atoms.get_indices())

            if len(atom_indices_in_negative_subspace) > 0 and len(atom_indices_in_positive_subspace) > 0:
                break

        new_particle = NP.Nanoparticle(common_lattice)
        new_particle.add_atoms(particle1.get_atoms(atom_indices_in_positive_subspace))
        new_particle.add_atoms(particle2.get_atoms(atom_indices_in_negative_subspace))

        new_particle.construct_bounding_box()

        n_atoms_old = particle1.get_n_atoms(include_X=False)
        new_particle.enforce_atom_number(n_atoms_old, self.fill_symbol)
        return new_particle


