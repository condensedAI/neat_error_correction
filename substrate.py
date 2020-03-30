from toric_game_env import Board

class Substrate(object):
    def __init__(self, size):
        board =  Board(size)

        # Input coordinates represent qubits and plaquette locations
        # Respect the board order for the perspective to still work
        self.input_coordinates = []
        for x in range(2*size):
            for y in range(2*size):
                if [x,y] in board.qubit_pos + board.plaquet_pos:
                    # Normalized coordinates
                    self.input_coordinates.append([(x - size)/size, (y - size)/size])

        # Output coordinates represent qubit flip action location
        a=1/size
        self.output_coordinates = [[a, 0], [0, -a], [-a, 0], [0, a]]

        # TODO: handle adding an extra hidden layer
        # In the form of 2d array, axis=0 is layer, axis=1 is neuron
        #self.hidden_coordinates = []
