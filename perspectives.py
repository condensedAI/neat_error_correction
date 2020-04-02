import numpy as np
from toric_game_env import Board

class Perspectives():
    def __init__(self, board_size, remove_star_op):
        self.size = board_size

        if remove_star_op:
            # Mask array of the star operator locations in the board array
            b = Board(board_size)
            mask_star_operators = [not [x,y] in b.star_pos for x in range(2*board_size) for y in range(2*board_size)]

        # Generate rotation symmetries
        # To make sure the lattice is in the right convention
        # Having first row and first column having star operators
        # We sometimes need to shift it by one row or column
        rolling_axis_after_rotation=[[], [0], [0,1], [1]]

        # Define the board_size**2 ways of shifting the board
        indices = np.arange(4*self.size**2, dtype=np.int16).reshape((2*self.size, 2*self.size))
        self.perspectives = {i : {} for i in range(4)}
        for rot_i in range(4):
            for i in range(board_size):
                for j in range(board_size):
                    # Shift the syndrome to central plaquette
                    plaq = [2*i+1, 2*j+1]
                    transformed_indices = np.roll(indices, (self.size - plaq[0])%(2*self.size), axis=0)
                    transformed_indices = np.roll(transformed_indices, (self.size - plaq[1])%(2*self.size), axis=1)

                    # Rotate
                    tranformed_indices = np.rot90(transformed_indices, rot_i)

                    # Shift again to replace the syndrome to central plaquette
                    for axis in rolling_axis_after_rotation[rot_i]:
                        tranformed_indices = np.roll(tranformed_indices, 1, axis)

                    if remove_star_op:
                        self.perspectives[rot_i][tuple(plaq)] = tranformed_indices.flatten()[mask_star_operators]
                    else:
                        self.perspectives[rot_i][tuple(plaq)] = tranformed_indices.flatten()

    # Return the indices of the new lattice shifted such that the syndrome is placed at the central plaquette
    # Also allows for returning the reflected indices given by rotation_number
    def shift_from(self, plaq, rotation_number=0):
        return self.perspectives[rotation_number][(plaq[0], plaq[1])]
