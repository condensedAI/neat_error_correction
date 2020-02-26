from __future__ import division
import numpy as np

import gym
from gym import spaces
from gym import error
from gym.utils import seeding
import sys, os
import copy

### Environment
class ToricGameEnv(gym.Env):
    '''
    ToricGameEnv environment. Effective single player game.
    '''
    metadata = {"render.modes": ["human", "ansi"]}

    def init(self, board_size, error_rate = 0.01):
        """
        Args:
            opponent: Fixed
            board_size: board_size of the board to use
        """
        self.board_size = board_size
        self.error_rate = error_rate

        self.seed()

        # opponent
        self.opponent_policy = None
        self.opponent = "OnceAtStart"

        # Observation space on board
        # The board size is defined by the number of plaquettes d
        # For the toric code, there are also d**2 qubits (identifying periodic)
        # So the observation space is 2*d**2. But that's hard to implement,
        # hence we go for 4*d**2, and ignore some
        shape = (2*self.board_size, 2*self.board_size) # board_size * board_size
        self.observation_space = spaces.Box(np.zeros(shape), np.ones(shape))

        # Keep track of the moves
        self.moves = []
        # Empty State
        self.state = Board(self.board_size)
        # reset the board during initialization
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**32
        return [seed1, seed2]

    def reset(self):
        # Let the opponent do it's initial evil
        self.state.reset()
        self.moves = []
        self._set_initial_errors()
        # Store the moves so that we can add them at the end
        self.initialmoves = copy.copy(self.moves)

        # reset action_space
        self.actions = []

        self.done = self.state.is_terminal()
        self.reward = 0
        if self.done:
            self.reward = 1
            if self.state.has_logical_error(self.initialmoves):
                self.reward = -1

        return self.state.encode()

    def close(self):
        self.opponent_policy = None
        self.state = None

    def render(self, mode="human", close=False):
        print("Not yet implemented")

    def step(self, action):
        '''
        Args:
            action: int
        Return:
            observation: board encoding,
            reward: reward of the game,
            done: boolean,
            info: state dict
        Raise:
            Illegal Move action, basically the position on board is not empty
        '''
        # If already terminal, then don't do anything, count as win
        if self.done:
            self.reward = 1
            return self.state.encode(), 1., True, {'state': self.state, 'message':"No errors."}

        self.actions.append(action)

        try:
            self.state.act(action)
        except error.Error:
            return self.state.encode(), -1.0, True, {'state': self.state, 'message': "Illegal action."}

        # Reward: if nonterminal, then the reward is 0
        if not self.state.is_terminal():
            self.done = False
            self.reward = 0
            return self.state.encode(), 0., False, {'state': self.state, 'message':"Continue"}

        # We're in a terminal state. Reward is 1 if won, -1 if lost
        self.done = True
        self.reward = -1.0 if self.state.has_logical_error(self.initialmoves) else 1.0
        return self.state.encode(), self.reward, True, {'state': self.state, 'message':"No syndrome."}

    def _set_initial_errors(self):
        ''' Set random initial errors
            but report only the syndrome
        '''
        # Pick random sites according to error rate
        #print("Setting initial errors")
        for i,m in enumerate(self.state.qubit_pos):
            if np.random.rand() < self.error_rate:
                self.moves.append( i )
                self.state.act(self.state.action_to_coord(i), init=True)

        # Now unflip the qubits, they're a secret
        for q in self.state.qubit_pos:
            self.state.board_state[q[0],q[1]] = 0

        self.state.move = 0
        self.state.last_coord = (-1,-1)     # last action coord
        self.state.last_action = None       # last action made


class Board(object):
    '''
    Basic Implementation of a ToricGame Board, actions are int [0,2*board_size**2)
    x : spin
    P : plaquette operator
    o : star operator

    o--x---o---x---o---x---
    |      |       |
    x  P   x   P   x   P
    |      |       |
    o--x---|---x---|---x---
    |      |       |
    x  P   x   P   x   P
    |      |       |
    o--x---|---x---|---x---
    |      |       |
    x  P   x   P   x   P
    |      |       |
    '''

    def __init__(self, board_size):
        self.size = board_size

        self.qubit_pos   = [[x,y] for x in range(2*self.size) for y in range((x+1)%2, 2*self.size, 2)]
        self.plaquet_pos = [[x,y] for x in range(1,2*self.size,2) for y in range(1,2*self.size,2)]
        self.star_pos    = [[x,y] for x in range(0,2*self.size,2) for y in range(0,2*self.size,2)]

        self.reset()

    def reset(self):
        self.board_state = np.zeros( (2*self.size, 2*self.size) )

        self.syndrome_pos = []

        self.move = 0                 # how many move has been made
        self.last_coord = (-1,-1)     # last action coord
        self.last_action = None       # last action made

    def action_to_coord(self, a):
        return self.qubit_pos[a]

    def isAdjacentToSyndrome(self, move):
        if self.board_state[(move[0]-1)%(2*self.size),move[1]] == 1:
            return True
        if self.board_state[(move[0]+1)%(2*self.size),move[1]] == 1:
            return True
        if self.board_state[move[0],(move[1]-1)%(2*self.size)] == 1:
            return True
        if self.board_state[move[0],(move[1]+1)%(2*self.size)] == 1:
            return True

        return False

    def get_legal_move(self):
        ''' Check all valid move positions (i.e. unflipped qubits)
            Return: Coordinate of all the empty space, [(x1, y1), (x2, y2), ...]
        '''
        legal_move = []
        for qp in self.qubit_pos:
            if (self.board_state[qp[0],qp[1]] == 0) and self.isAdjacentToSyndrome(qp):
                legal_move.append( qp )
        return legal_move

    def get_legal_action(self):
        ''' Check all valid actions (i.e. qubits that can be flipped)
            Return: Actions belonging to the empty spaces
        '''
        legal_action = []
        for i,qp in enumerate(self.qubit_pos):
            if (self.board_state[qp[0],qp[1]] == 0) and self.isAdjacentToSyndrome(qp):
                legal_action.append( i )
        return legal_action

    def act(self, action, init=False, allow_reflip=False):
        '''
            Args: input action in the form of position [x,y]
        '''
        # Get the qubit to flip, and check if valid
        coord = action
        #print(coord)

        # check if the qubit has already been flipped
        if (self.board_state[coord[0]][coord[1]] != 0) and not allow_reflip:
            raise error.Error("Action is illegal, position [%d, %d] already flipped" % ((coord[0]),(coord[1])))

        # Flip it!
        self.board_state[coord[0],coord[1]] = (self.board_state[coord[0], coord[1]] + 1) % 2
        self.move += 1 # move counter add 1
        self.last_coord = coord # save last coordinate
        self.last_action = action

        # Update the syndrome measurements
        # Only need to incrementally change
        # Find plaquettes that the flipped qubit is a part of
        # Only flips plaquettes operators (no star ones)
        if coord[0] % 2 == 0:
            plaqs = [ [ (coord[0] + 1) % (2*self.size), coord[1] ], [ (coord[0] - 1) % (2*self.size), coord[1] ] ]
        else:
            plaqs = [ [ coord[0], (coord[1] + 1) % (2*self.size) ], [ coord[0], (coord[1] - 1) % (2*self.size) ] ]


        # Update syndrome positions
        for plaq in plaqs:
            if plaq in self.syndrome_pos:
                self.syndrome_pos.remove(plaq)
            else:
                self.syndrome_pos.append(plaq)

            self.board_state[plaq[0],plaq[1]] = (self.board_state[plaq[0],plaq[1]] + 1) % 2


    def is_terminal(self):
        # Check if there are valid moves left
        if len(self.get_legal_action()) == 0:
            return True

        # Are all syndromes removed?
        plaqs = [ self.board_state[p] for p in self.plaquet_pos ]
        num_syndromes = np.sum( plaqs)
        if num_syndromes != 0:
            return False

        # There are moves left, but no more syndromes; so game done
        return True

    def copy(self, board_state):
        ''' Update board_state of current board values from input 2D list
        '''
        for i in range(2*self.size):
            for j in range(2*self.size):
                self.board_state[i][j] = board_state[i][j]

    def has_logical_error(self, initialmoves, debug=False):
        # Add the original zerrors
        revealed_board = Board(self.size)
        revealed_board.copy( self.board_state )

        # Allow flipping of already flipped qubits, but only here
        for m in initialmoves:
            revealed_board.act(self.action_to_coord(m), allow_reflip=True, init=True)

        if debug:
            print("## DEBUG INFO: \n")
            print("The board with initialmoves: \n")
            print(revealed_board.board_state)
            print("\n")

        # Check for Z error
        z1pos = [[0,x] for x in range(1, 2*self.size, 2)]
        z2pos = [[y,0] for y in range(1, 2*self.size, 2)]
        #x1pos = [[1,x] for x in range(0, 2*self.size, 2)]
        #x2pos = [[y,1] for y in range(0, 2*self.size, 2)]

        if debug:
            print("## DEBUG INFO: \n")
            print("State of qubits: ")
            print( [revealed_board.board_state[p[0], p[1]] for p in z1pos ] )
            print( [revealed_board.board_state[p[0], p[1]] for p in z2pos ] )

        zerrors = [0,0]
        for pos in z1pos:
            zerrors[0] += revealed_board.board_state[ pos[0], pos[1] ]
        for pos in z2pos:
            zerrors[1] += revealed_board.board_state[ pos[0], pos[1] ]

        if (zerrors[0]%2 == 1) or (zerrors[1]%2 == 1):
            return True

        return False

    def __repr__(self):
        ''' representation of the board class
            print out board_state
        '''
        out = self.board_state
        return out

    def encode(self):
        '''Return: np array
            np.array(board_size, board_size): state observation of the board
        '''
        img = np.array(self.board_state)
        return img
