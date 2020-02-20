import numpy as np


class BoundingBox:
    def __init__(self):
        self.width = 0
        self.length = 0
        self.height = 0

        self.position = np.array([0, 0, 0])

    def get_center(self):
        return self.position + np.array([self.width/2, self.length/2, self.height/2])

    def construct(self, lattice, atom_indices):
        min_coordinates = np.array([1e10, 1e10, 1e10])
        max_coordinates = np.array([-1e10, -1e10, -1e10])

        for lattice_index in atom_indices:
            cur_position = lattice.get_cartesian_position_from_index(lattice_index)
            for coordinate in range(3):
                if cur_position[coordinate] < min_coordinates[coordinate]:
                    min_coordinates[coordinate] = cur_position[coordinate]
                if cur_position[coordinate] > max_coordinates[coordinate]:
                    max_coordinates[coordinate] = cur_position[coordinate]

        self.width = max_coordinates[0] - min_coordinates[0]
        self.length = max_coordinates[1] - min_coordinates[1]
        self.height = max_coordinates[2] - min_coordinates[2]
        self.position = min_coordinates

    def print(self):
        print('Bounding Box:')
        print("width: {0} length: {1} height: {2}".format(self.width, self.length, self.height))
        print("AnchorPoint: x = {0}, y = {1}, z = {2}".format(self.position[0], self.position[1], self.position[2]))
        print("Center: {0}".format(self.get_center()))
