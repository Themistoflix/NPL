from Core import Nanoparticle as NP

class CutAndSpliceOperator:
    def __init__(self, cutting_plane_generator):
        self.cutting_plane_generator = cutting_plane_generator

    def cut_and_splice(self, particle1, particle2):
        self.cutting_plane_generator.setCenter(particle1.boundingBox.get_center())
        common_lattice = particle1.lattice

        # make sure that we actually cut
        while True:
            cutting_plane = self.cutting_plane_generator.generateNewCuttingPlane()
            atom_indices_in_positive_subspace, _ = cutting_plane.splitAtomIndices(common_lattice, particle1.atoms.getIndices())
            _, atom_indices_in_negative_subspace = cutting_plane.splitAtomIndices(common_lattice, particle2.atoms.getIndices())

            if len(atom_indices_in_negative_subspace) > 0 and len(atom_indices_in_positive_subspace) > 0:
                break

        new_particle = NP.Nanoparticle(common_lattice)
        new_particle.add_atoms(particle1.get_atoms(atom_indices_in_positive_subspace))
        new_particle.add_atoms(particle2.get_atoms(atom_indices_in_negative_subspace))

        new_particle.construct_bounding_box()

        # old_stoichiometry = particle1.getStoichiometry()
        # new_particle.enforceStoichiometry(old_stoichiometry)
        return new_particle


