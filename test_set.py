import numpy as np
import sys

from game import ToricCodeGame
from simple_feed_forward import SimpleFeedForwardNetwork
from config import ErrorMode, GameMode, RewardMode

class TestSet():
    def __init__(self, board_size, error_rates, n_games, max_steps):
        self.error_rates = error_rates
        self.n_games = n_games

        self.game = ToricCodeGame(board_size, max_steps, epsilon=0)

        # Random seeds of the test set
        maxseed=2**32-1
        self.sample_seeds = [np.random.randint(0, maxseed) for i in range(n_games)]

    def evaluate(self, genome, config):
        net = SimpleFeedForwardNetwork.create(genome, config)
        fitness = 0
        for i in range(self.n_games):
            # Determine the initial random error configuration
            np.random.seed(self.sample_seeds[i])

            # Create puzzles of varying difficulties
            error_rate = self.error_rates[i%len(self.error_rates)]
            fitness += self.game.play(net, error_rate, ErrorMode["PROBABILISTIC"], RewardMode["BINARY"], GameMode["TRAINING"])["fitness"]

        return fitness / self.n_games
