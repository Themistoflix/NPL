import numpy as np


class FCCLattice:
    # Use a right-handed cartesian coordinate system with only positive coordinates
    # width  is associated with the x component
    # length is associated with the y component
    # height is associated with the z component
    MAX_NEIGHBORS = 12

    def __init__(self, width, length, height, latticeConstant):
        assert width % 2 == 1 and length % 2 == 1 and height % 2 == 1, "box dimensions need to be odd!"
        self.width = width
        self.length = length
        self.height = height
        self.lattice_constant = latticeConstant

        self.lattice_points_per_even_layer = (int(self.width / 2) + 1) ** 2 + (int(self.width / 2)) ** 2
        self.lattice_points_per_odd_layer = (width - 1) / 2 * (length + 1) / 2 + (width + 1) / 2 * (length - 1) / 2

    def is_valid_index(self, index):
        if (index < 0) or (index > (self.height + 1) / 2 * self.lattice_points_per_even_layer + (
                self.height - 1) / 2 * self.lattice_points_per_odd_layer):
            return False

        return True

    def is_valid_lattice_position(self, position):
        # check boundaries
        if position[0] < 0 or position[1] < 0 or position[2] < 0:
            return False

        if position[0] > self.width or position[1] > self.length or position[2] > self.height:
            return False

        # check if the position specifies as lattice point
        if position[2] % 2 == 0:
            if (position[0] % 2 == 0 and position[1] % 2 == 0) or (position[0] % 2 == 1 and position[1] % 2 == 1):
                return True
            else:
                return False
        if position[2] % 2 == 1:
            if (position[0] % 2 == 0 and position[1] % 2 == 1) or (position[0] % 2 == 1 and position[1] % 2 == 0):
                return True
            else:
                return False

    def get_index_from_lattice_position(self, position):
        def index_in_xy_plane(x, y, z, width):
            if z % 2 == 0:
                if x % 2 == 0:
                    return y / 2 * (width + 1) / 2 + (y / 2) * (width - 1) / 2 + (x / 2)
                else:
                    return (y + 1) / 2 * (width + 1) / 2 + (y - 1) / 2 * (width - 1) / 2 + (x - 1) / 2
            else:
                if x % 2 == 0:
                    return (y + 1) / 2 * (width - 1) / 2 + (y - 1) / 2 * (width + 1) / 2 + (x / 2)
                else:
                    return y / 2 * (width + 1) / 2 + y / 2 * (width - 1) / 2 + (x - 1) / 2

        odd_layers_below = 0
        even_layers_below = 0
        if position[2] % 2 == 0:
            odd_layers_below = even_layers_below = position[2] / 2
        else:
            odd_layers_below = (position[2] - 1) / 2
            even_layers_below = (position[2] + 1) / 2
        index = odd_layers_below * self.lattice_points_per_odd_layer + even_layers_below * self.lattice_points_per_even_layer \
                + index_in_xy_plane(position[0], position[1], position[2], self.width)

        return index

    def get_index_from_cartesian_position(self, position):
        return self.get_index_from_lattice_position(position / self.lattice_constant)

    def get_lattice_position_from_index(self, index):
        def position_from_plane_index(plane_index, z, width):
            # a 'full line' is a row with length l and the following row with length l - 1 (or l + 1)
            full_lines_above = int(plane_index / width)
            # shift the index into the first line
            new_index = plane_index - full_lines_above * width
            if int(z) % 2 == 0:
                # check in which row the index is
                x = y = 0
                if new_index >= (width + 1) / 2:
                    y = 1
                else:
                    y = 0

                if int(y) == 0:  # upper row
                    x = new_index * 2
                else:  # lower row
                    new_index = new_index - (width + 1) / 2
                    x = new_index * 2 + 1

            if int(z) % 2 == 1:
                # check in which row the index is
                if new_index < (width - 1) / 2:
                    y = 0
                else:
                    y = 1

                if y == 0:
                    x = new_index * 2 + 1
                else:
                    new_index = new_index - (width - 1) / 2
                    x = new_index * 2

            return x, y + 2 * full_lines_above

        # a 'full block' is an even and an odd layer
        full_blocks_below = int(index / (self.lattice_points_per_even_layer + self.lattice_points_per_odd_layer))
        # shift the index into the first line
        new_index = index - full_blocks_below * (self.lattice_points_per_even_layer + self.lattice_points_per_odd_layer)

        # check if it is in the upper (odd) or lower(even) layer
        if new_index >= self.lattice_points_per_even_layer:
            z = 1
        else:
            z = 0
        plane_index = new_index - z * self.lattice_points_per_even_layer
        x, y = position_from_plane_index(plane_index, z, self.width)

        return np.array([x, y, z + 2 * full_blocks_below])

    def get_cartesian_position_from_index(self, index):
        return self.get_lattice_position_from_index(index) * self.lattice_constant

    def get_anchor_index_of_centered_box(self, w, h, l):
        anchor_point_x = int((self.width - w - 1) / 2)
        anchor_point_y = int((self.length - l - 1) / 2)
        anchor_point_z = int((self.height - h - 1) / 2)

        if not self.is_valid_lattice_position(np.array([anchor_point_x, anchor_point_y, anchor_point_z])):
            anchor_point_z = anchor_point_z + 1

        return np.array([anchor_point_x, anchor_point_y, anchor_point_z])

    def get_nearest_neighbors(self, index):
        position = self.get_lattice_position_from_index(index)

        neighbors = set()
        for xOffset in [-1, 0, 1]:
            for yOffset in [-1, 0, 1]:
                for zOffset in [-1, 0, 1]:
                    if xOffset is yOffset is zOffset is 0:
                        continue
                    offset = np.array([xOffset, yOffset, zOffset])

                    if self.is_valid_lattice_position(position + offset):
                        neighborIndex = self.get_index_from_lattice_position(position + offset)
                        neighbors.add(neighborIndex)

        return neighbors
