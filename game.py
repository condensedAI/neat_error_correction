
from scipy.special import softmax
import numpy as np
import time
import neat

from toric_game_env import ToricGameEnv
from perspectives import Perspectives

from config import GameMode

class ToricCodeGame():
    def __init__(self, board_size, max_steps, epsilon):
        self.board_size = board_size
        self.max_steps = max_steps
        self.epsilon = epsilon

        self.env = ToricGameEnv(self.board_size)

        # The perspective includes the masking of star operators
        self.perspectives = Perspectives(self.board_size, self.env.state.star_pos)

    # Return the score of the game
    # In evaluation mode, the fitness is in {0, 1} corresponding to success or failure
    # In training mode, fitness can be defined differently
    def play(self, nn, error_rate, error_mode, mode, verbose=False):

        current_state = self.env.generate_errors(error_rate, error_mode)

        # If there is no syndrome in the initial configuration
        # Either we generate a new one containing syndromes
        # Or if there happens to be a logical error, we return a failure
        if mode == GameMode["TRAINING"]:
            while self.env.done and error_rate>0:
                if self.env.reward == -1:
                    # (Since it is not possible to correct these errors, there is no reward nor penalty)
                    return {"fitness":0, "error_rate": error_rate, "outcome":"logical_error", "nsteps":0}

                current_state = self.env.generate_errors(error_rate, error_mode)

        # In evaluation mode, we keep even these empty initial configurations
        elif self.env.done:
            if self.env.reward == -1:
                return {"fitness":0, "error_rate": error_rate, "outcome":"logical_error", "nsteps":0}
            else:
                return {"fitness":1, "error_rate": error_rate, "outcome":"success", "nsteps":0}

        if verbose:
            print("Initial", current_state)
            print(self.env.done, self.env.state.syndrome_pos)
            if verbose > 1: self.env.render()

        for step in range(self.max_steps+1):

            current_state = current_state.flatten()

            actions=[]
            probs=[]
            # Go over perspectives
            for plaq in self.env.state.syndrome_pos:
                # Shift the board to center the syndrome
                # Also includes masking of star operators
                indices = self.perspectives.shift_from(plaq)

                input = current_state[indices]

                probs += list(nn.activate(input))
                # Output has 4 neurons
                actions += [[(plaq[0]+1)%(2*self.board_size), plaq[1]]]
                actions += [[(plaq[0]-1)%(2*self.board_size), plaq[1]]]
                actions += [[plaq[0], (plaq[1]+1)%(2*self.board_size)]]
                actions += [[plaq[0], (plaq[1]-1)%(2*self.board_size)]]

            # To avoid calling rand() when evaluating (for testing purposes)
            if self.epsilon == 0 or mode == GameMode["EVALUATION"]:
                action=actions[np.argmax(probs)]
            else:
                # epsilon-greedy search
                if np.random.rand() < self.epsilon:
                    action = actions[np.random.randint(len(actions))]
                else:
                    action = actions[np.argmax(probs)]

            current_state, reward, done, info = self.env.step(action)

            if verbose:
                print(step, current_state, reward, action, info["message"])
                if verbose > 1: self.env.render()

            # if no syndromes are present anymore
            if done:
                # Fitness is 1 if there is no logical error
                # 0 if there is a logical error
                if mode == GameMode["TRAINING"]:
                    # The harder the puzzle the higher the reward and the lower the penalty
                    # The easier the puzzles the lower the reward and the higher the penalty
                    #fitness = (reward-1)/2 + len(self.env.initial_qubits_flips)/(2*self.board_size**2)
                    fitness = (reward+1)/2
                    return {"fitness": fitness, "error_rate": error_rate, "outcome":info["message"], "nsteps":step}
                else:
                    return {"fitness":(reward+1)/2, "error_rate": error_rate, "outcome":info["message"], "nsteps":step}

        # When the number of moves went beyond max_steps
        if mode == GameMode["TRAINING"]:
            # The harder the puzzle the higher the reward and the lower the penalty
            # The easier the puzzles the lower the reward and the higher the penalty
            #fitness = (reward-1)/2 + len(self.env.initial_qubits_flips)/(2*self.board_size**2)
            fitness = 0
            return {"fitness": fitness, "error_rate": error_rate, "outcome":info["message"], "nsteps":step}
        else:
            return {"fitness":0, "error_rate": error_rate, "outcome":"max_steps", "nsteps":max_step}


    def close(self):
        self.env.close()
