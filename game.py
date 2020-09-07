
from scipy.special import softmax
import numpy as np
import time
import neat

from toric_game_env import ToricGameEnv
from perspectives import Perspectives

from config import GameMode, RewardMode

class ToricCodeGame():
    def __init__(self, config):
        self.board_size = config["Physics"]["distance"]
        self.max_steps = config["Training"]["max_steps"]
        self.epsilon = config["Training"]["epsilon"]
        self.network_type = config["Training"]["network_type"]
        self.substrate_type = config["Training"]["substrate_type"]
        self.rotation_invariant_decoder = config["Training"]["rotation_invariant_decoder"]

        self.env = ToricGameEnv(self.board_size)

        # Very important to change the seed here
        # Otherwise for game evaluation in parallel
        # All game objects will share the same seed leading to biased results
        np.random.seed()

        # The perspective includes the masking of star operators
        self.perspectives = Perspectives(self.board_size,
                                        remove_star_op=True,
                                        remove_qubits=not config["Training"]["memory"])

    # Return the score of the game
    # In evaluation mode, the fitness is in {0, 1} corresponding to success or failure
    # In training mode, fitness can be defined differently
    def play(self, nn, error_rate, error_mode, reward_mode, mode, seed=None, verbose=False):
        #if not seed is None:
        #    np.random.seed(seed)

        current_state = self.env.generate_errors(error_rate, error_mode)

        # If there is no syndrome in the initial configuration
        # Either we generate a new one containing syndromes
        # Or if there happens to be a logical error, we return a failure
        if mode == GameMode["TRAINING"]:
            while self.env.done and error_rate>0:
                if self.env.reward == -1:
                    # In both reward modes BINARY or CURRICULUM, since it is not possible to correct these errors, there is no reward nor penalty
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

            if self.network_type == 'cppn' and self.substrate_type == 1:
                probs, actions = self._get_actions(nn, current_state)
            else:
                if not self.rotation_invariant_decoder:
                    probs, actions = self._get_actions_with_perspectives(nn, current_state)
                else:
                    probs, actions = self._get_actions_with_perspectives_rotation_invariant(nn, current_state)

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
                    if reward_mode == RewardMode["BINARY"]:
                        fitness = (reward+1)/2
                    elif reward_mode == RewardMode["CURRICULUM"]:
                        fitness = (reward-1)/2 + len(self.env.initial_qubits_flips)/(2*self.board_size**2)

                    return {"fitness": fitness, "error_rate": error_rate, "outcome":info["message"], "nsteps":step}
                else:
                    return {"fitness":(reward+1)/2, "error_rate": error_rate, "outcome":info["message"], "nsteps":step}

        # When the number of moves went beyond max_steps
        if mode == GameMode["TRAINING"]:
            # The harder the puzzle the higher the reward and the lower the penalty
            # The easier the puzzles the lower the reward and the higher the penalty
            if reward_mode == RewardMode["BINARY"]:
                fitness = 0
            elif reward_mode == RewardMode["CURRICULUM"]:
                fitness = -1 + len(self.env.initial_qubits_flips)/(2*self.board_size**2)
            return {"fitness": fitness, "error_rate": error_rate, "outcome":info["message"], "nsteps":step}
        else:
            return {"fitness":0, "error_rate": error_rate, "outcome":"max_steps", "nsteps":max_step}


    # In this case, the output layer has 4 nodes
    def _get_actions_with_perspectives(self, nn, current_state):
        # Go over perspectives
        actions, probs=[], []
        for plaq in self.env.state.syndrome_pos:
            # Shift the board to center the syndrome
            # Also includes masking of star operators
            indices = self.perspectives.shift_from(plaq)

            input = current_state[indices]

            # NN outputs 4 values
            probs += list(nn.activate(input))

            # 4 possible actions
            # Bad convention for action order, but we must keep it for retrocompatibility
            actions += [[(plaq[0]+1)%(2*self.board_size), plaq[1]]]
            actions += [[(plaq[0]-1)%(2*self.board_size), plaq[1]]]
            actions += [[plaq[0], (plaq[1]+1)%(2*self.board_size)]]
            actions += [[plaq[0], (plaq[1]-1)%(2*self.board_size)]]

        return probs, actions

    # In this case, the output layer has 1 node
    def _get_actions_with_perspectives_rotation_invariant(self, nn, current_state):

        # Go over syndromes
        actions, probs=[], []
        for plaq in self.env.state.syndrome_pos:
            # Rotation  order corresponding the 90degree anti-clockwise rotation
            qubit_flips_rotationally_ordered=[]
            qubit_flips_rotationally_ordered += [[(plaq[0]+1)%(2*self.board_size), plaq[1]]]
            qubit_flips_rotationally_ordered += [[plaq[0], (plaq[1]-1)%(2*self.board_size)]]
            qubit_flips_rotationally_ordered += [[(plaq[0]-1)%(2*self.board_size), plaq[1]]]
            qubit_flips_rotationally_ordered += [[plaq[0], (plaq[1]+1)%(2*self.board_size)]]

            for rot_i in range(4):
                indices = self.perspectives.shift_from(plaq, rot_i)

                input = current_state[indices]

                # NN outputs 1 value
                probs += list(nn.activate(input))

                actions += [qubit_flips_rotationally_ordered[rot_i]]

        return probs, actions

    # In this case, output layer has 2*board_size**2 (# of spins) nodes
    def _get_actions(self, nn, current_state):
        indices = self.perspectives.mask_star_operators

        input = current_state[indices]

        probs = list(nn.activate(input))

        actions = self.env.state.qubit_pos

        return probs, actions

    def close(self):
        self.env.close()
