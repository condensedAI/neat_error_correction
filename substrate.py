from toric_game_env import Board

class Substrate(object):
    def __init__(self, size):
        board =  Board(size)

        # Input coordinates represent qubits and plaquette locations
        # Respect the board order for the perspective to still work
        self.input_coordinates = []
        self.input_type = []
        for x in range(2*size):
            for y in range(2*size):
                if [x,y] in board.qubit_pos + board.plaquet_pos:
                    # Normalized coordinates
                    self.input_coordinates.append([(x - size)/size, (y - size)/size])
                    if [x,y] in board.qubit_pos:
                        self.input_type.append("Q")
                    elif [x,y] in board.plaquet_pos:
                        self.input_type.append("P")

        # Output coordinates represent qubit flip action location
        # Important: must correspond to the Game class interpretation of output
        a=1/size
        self.output_coordinates = [[a, 0], [-a, 0], [0, a], [0, -a]]

        # TODO: handle adding an extra hidden layer
        # In the form of 2d array, axis=0 is layer, axis=1 is neuron
        #self.hidden_coordinates = []
