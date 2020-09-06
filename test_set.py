import numpy as np
import sys
from multiprocessing import Pool

from game import ToricCodeGame
from simple_feed_forward import SimpleFeedForwardNetwork
from config import ErrorMode, GameMode, RewardMode
from phenotype_network import PhenotypeNetwork
from substrates import *
from neat.nn import FeedForwardNetwork

class TestSet():
    def __init__(self, config, n_games):
        self.board_size = config["Physics"]["distance"]
        self.max_steps = config["Training"]["max_steps"]
        self.error_rates = config["Training"]["error_rates"]
        self.n_games = n_games
        self.network_type = config["Training"]["network_type"]

        if self.network_type == 'cppn':
            if config["Training"]["substrate_type"] == 0:
                self.substrate = SubstrateType0(config["Physics"]["distance"], config["Training"]["rotation_invariant_decoder"])
            if config["Training"]["substrate_type"] == 1:
                self.substrate = SubstrateType1(config["Physics"]["distance"])

        self.game = ToricCodeGame(config)
        self.game.epsilon = 0

        # Random seeds of the test set
        maxseed=2**32-1
        self.sample_seeds = [np.random.randint(0, maxseed) for i in range(n_games)]

    def evaluate(self, pool_workers, genome, config):
        if self.network_type == 'ffnn':
            net = SimpleFeedForwardNetwork.create(genome, config)
        elif self.network_type == 'cppn':
            cppn_network = FeedForwardNetwork.create(genome, config)
            net = PhenotypeNetwork.create(cppn_network, self.substrate)

        jobs=[]
        for i in range(self.n_games):
            # Create puzzles of varying difficulties
            error_rate = self.error_rates[i%len(self.error_rates)]
            # Determine the initial random error configuration
            seed = self.sample_seeds[i]
            jobs.append(pool_workers.apply_async(self.get_fitness, (net, error_rate, seed)))

        fitness = {err: 0 for err in self.error_rates}
        for job in jobs:
            res, error_rate = job.get(timeout=None)
            fitness[error_rate] += res

        total_fitness = sum(fitness.values()) / self.n_games
        details = [fitness[err]*len(self.error_rates)/ self.n_games for err in self.error_rates]
        return total_fitness, details

    def get_fitness(self, net, error_rate, seed):
        results = self.game.play(net, error_rate, ErrorMode["PROBABILISTIC"], RewardMode["BINARY"], GameMode["TRAINING"], seed)
        return results["fitness"], results['error_rate']
        """
        def evaluate(self, pool_workers, genome, config):
            net = SimpleFeedForwardNetwork.create(genome, config)
            fitness = 0
            jobs=[]
            for i in range(self.n_games):

                # Determine the initial random error configuration
                np.random.seed(self.sample_seeds[i])

                # Create puzzles of varying difficulties
                error_rate = self.error_rates[i%len(self.error_rates)]
                fitness += self.game.play(net, error_rate, ErrorMode["PROBABILISTIC"], RewardMode["BINARY"], GameMode["TRAINING"])["fitness"]

            return fitness / self.n_games
        """
