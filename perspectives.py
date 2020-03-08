import numpy as np

class Perspectives():
    def __init__(self, board_size, star_pos=None):
        self.size = board_size

        if not star_pos is None:
        # Mask array of the star operator locations in the board array
            mask_star_operators = [not [x,y] in star_pos for x in range(2*board_size) for y in range(2*board_size)]

        # Define the board_size**2 ways of shifting the board
        indices = np.arange(4*self.size**2)
        indices = np.reshape(indices, (2*self.size, 2*self.size))
        self.perspectives = {}
        for i in range(board_size):
            for j in range(board_size):
                plaq = [2*i+1, 2*j+1]
                shifted_indices = np.roll(indices, (self.size - plaq[0])%(2*self.size), axis=0)
                shifted_indices = np.roll(shifted_indices, (self.size - plaq[1])%(2*self.size), axis=1)

                if star_pos is None:
                    self.perspectives[tuple(plaq)] = shifted_indices.flatten()[mask_star_operators]
                else:
                    self.perspectives[tuple(plaq)] = shifted_indices.flatten()[mask_star_operators]
                
    def shift_from(self, plaq):
        return self.perspectives[(plaq[0], plaq[1])]
