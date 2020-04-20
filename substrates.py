from toric_game_env import Board

class SubstrateType0(object):
    '''
    This substrate consists of:
    - an input layer representing the board (plaquette and spin operators)
    - an output layer representing the 4 nearest-to-syndrome spin to flip
    This substrate should be used in conjunction with the translated perspectives
    '''
    def __init__(self, size, rotation_invariance):
        board =  Board(size)
        self.with_coord_diff = False

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
                        self.input_type.append("Q"+str(board.qubit_pos.index([x,y])))
                    elif [x,y] in board.plaquet_pos:
                        self.input_type.append("P"+str(board.plaquet_pos.index([x,y])))

        # Output coordinates represent qubit flip action location
        # Important: must correspond to the Game class interpretation of output
        a=1/size
        if rotation_invariance:
            #self.output_coordinates = [[0, -a]]
            self.output_coordinates = [[0, 0]]
        else:
            self.output_coordinates = [[a, 0], [-a, 0], [0, a], [0, -a]]

        # TODO: handle adding an extra hidden layer
        # In the form of 2d array, axis=0 is layer, axis=1 is neuron
        #self.hidden_coordinates = []

class SubstrateType1(object):
    '''
    This substrate consists of:
    - an input layer representing the board (L**2 plaquette and 2*L**2 spin operators)
    - an output layer representing all the spins 2*L**2
    In this case, the perspectives are not used and translation invariance should
    manifest in taking only coordinate difference for weight determination in CPPN
    '''
    def __init__(self, size):
        board =  Board(size)
        self.with_coord_diff = True

        # Input coordinates represent qubits and plaquette locations
        # Respect the board order for the perspective to still work
        self.input_coordinates, self.output_coordinates = [], []
        for x in range(2*size):
            for y in range(2*size):
                if [x,y] in board.qubit_pos:
                    # Normalized coordinates
                    self.input_coordinates.append([(x - size)/size, (y - size)/size])
                    self.output_coordinates.append([(x - size)/size, (y - size)/size])
                if [x,y] in board.plaquet_pos:
                    self.input_coordinates.append([(x - size)/size, (y - size)/size])

        # TODO: handle adding an extra hidden layer
        # In the form of 2d array, axis=0 is layer, axis=1 is neuron
        #self.hidden_coordinates = []
