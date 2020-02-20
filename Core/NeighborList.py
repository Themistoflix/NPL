class NeighborList:
    def __init__(self, lattice):
        self.list = dict()
        self.lattice = lattice

    def __getitem__(self, item):
        return self.list[item]

    def __setitem__(self, key, value):
        self.list[key] = value

    def construct(self, lattice_indices):
        for lattice_index in lattice_indices:
            nearest_lattice_neighbors = self.lattice.get_nearest_neighbors(lattice_index)
            nearest_neighbors = set()
            for neighbor in nearest_lattice_neighbors:
                if neighbor in lattice_indices:
                    nearest_neighbors.add(neighbor)

            self.list[lattice_index] = nearest_neighbors

    def add_atoms(self, lattice_indices):
        all_atoms = list(self.list.keys())
        for latticeIndex in lattice_indices:
            nearest_neighbors = set()
            nearest_lattice_neighbors = self.lattice.get_nearest_neighbors(latticeIndex)
            for neighbor in nearest_lattice_neighbors:
                if neighbor in all_atoms:
                    nearest_neighbors.add(neighbor)
                    self.list[neighbor].add(latticeIndex)

            self.list[latticeIndex] = nearest_neighbors

    def remove_atoms(self, lattice_indices):
        for lattice_index in lattice_indices:
            neighbors = self.list[lattice_index]
            for neighbor in neighbors:
                self.list[neighbor].remove(lattice_index)

            del self.list[lattice_index]

    def get_coordination_number(self, lattice_index):
        return len(self.list[lattice_index])
