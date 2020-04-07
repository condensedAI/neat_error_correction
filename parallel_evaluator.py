"""
Runs evaluation functions in parallel subprocesses
in order to evaluate multiple genomes at once.
"""

from abstract_parallel_evaluator import AbstractParallelEvaluator

from neat.nn import FeedForwardNetwork
from game import ToricCodeGame
from simple_feed_forward import SimpleFeedForwardNetwork
#from genome_conversion import convert_to_ANN_genome
from phenotype_network import PhenotypeNetwork
from substrates import *
from config import GameMode

# This is the object copied on each subprocess
# It contains the essential variables
class FitnessEvaluator(object):
    def __init__(self, config):
        self.error_rates = config["Training"]["error_rates"]
        self.error_mode = config["Training"]["error_mode"]
        self.reward_mode = config["Training"]["reward_mode"]
        self.n_games = config["Training"]["n_games"]
        self.network_type = config["Training"]["network_type"]

        if self.network_type == 'cppn':
            if config["Training"]["substrate_type"] == 0:
                self.substrate = SubstrateType0(config["Physics"]["distance"])
            if config["Training"]["substrate_type"] == 1:
                self.substrate = SubstrateType1(config["Physics"]["distance"])

        self.game = ToricCodeGame(config)
        
    def __del__(self):
        self.game.close()

    def get(self, genome, config):
        '''
        if self.network_type == 'ffnn':
            genome_nn = genome # No need to copy in this case, since there is no modification
        if self.network_type == 'cppn':
            # Transform the CPPN genome to an ANN genome
            genome_nn = convert_to_ANN_genome(genome, substrate, config)
        net = SimpleFeedForwardNetwork.create(genome_nn, config)
        '''
        if self.network_type == 'ffnn':
            net = SimpleFeedForwardNetwork.create(genome, config)
        elif self.network_type == 'cppn':
            cppn_net = FeedForwardNetwork.create(genome, config)
            net = PhenotypeNetwork.create(cppn_net, self.substrate)

        fitness = 0
        #print(id(self.game))
        for i in range(self.n_games):
            # Create puzzles of varying difficulties
            error_rate = self.error_rates[i%len(self.error_rates)]
            fitness += self.game.play(net, error_rate, self.error_mode, self.reward_mode, GameMode["TRAINING"])["fitness"]
        return fitness / self.n_games

# Regular inherits from ParallelEvaluatorParent
class ParallelEvaluator(AbstractParallelEvaluator):
    def __init__(self, num_workers, config, global_test_set=True, timeout=None):
        super().__init__(num_workers, config, global_test_set, timeout)

        # This is object copied on each core
        self.fitness_evaluator = FitnessEvaluator(config)

    def evaluate(self, genomes, config):
        jobs = []
        for ignored_genome_id, genome in genomes:
            #print(genome)
            jobs.append(self.pool.apply_async(self.fitness_evaluator.get, (genome, config)))

        # assign the fitness back to each genome
        best = None # Best genome of the generation
        for job, (ignored_genome_id, genome) in zip(jobs, genomes):
            genome.fitness = job.get(timeout=self.timeout)

            if best is None or genome.fitness > best.fitness:
                best = genome

        if self.global_test_set:
            self.test(best, config)
