"""
Runs evaluation functions in parallel subprocesses
in order to evaluate multiple genomes at once.
"""
from multiprocessing import Pool
import copy

from resampling_algorithm import ResamplingAlgorithm
from test_set import TestSet

# The existence of this class lies on the necessity
# to share the fail_counts dictionary for the resampling algorithm
# because we want to average results over the population at a given generation

class ParallelEvaluatorResampling(object):
    def __init__(self, num_workers, eval_function, config, timeout=None):
        """
        eval_function should take one argument, a tuple of
        (genome object, config object), and return
        a single float (the genome's fitness).
        """
        self.num_workers = num_workers
        self.eval_function = eval_function
        self.timeout = timeout

        self.pool = Pool(num_workers)

        self.resampling = ResamplingAlgorithm(config["Training"]["error_rates"], config["Population"]["pop_size"], config["Training"]["n_games"])

        # Keeping track of the best genome evaluated on the exact same test set
        self.best_genome = None
        self.best_genome_test_score = 0
        self.test_set = TestSet(config["Physics"]["distance"], \
                                config["Training"]["error_rates"], \
                                n_games=4000, \
                                max_steps=config["Training"]["max_steps"])

    def __del__(self):
        self.pool.close() # should this be terminate?
        self.pool.join()

    def evaluate(self, genomes, config):
        self.resampling.reset()

        jobs = []
        for ignored_genome_id, genome in genomes:
            jobs.append(self.pool.apply_async(self.eval_function, (genome, config, self.resampling.puzzles_proportions)))

        # assign the fitness back to each genome
        best = None # Best genome of the generation
        for job, (ignored_genome_id, genome) in zip(jobs, genomes):
            genome.fitness, detailed_results = job.get(timeout=self.timeout)
            for error_rate in self.resampling.error_rates:
                self.resampling.fail_count[error_rate] += detailed_results[error_rate]

            if best is None or genome.fitness > best.fitness:
                best = genome

        # Evaluate the best genome of the generation on the test set
        test_score = self.test_set.evaluate(best, config)
        if test_score > self.best_genome_test_score:
            # Make sure to do a deep copy
            self.best_genome = copy.copy(best)
            self.best_genome_test_score = test_score
            print("NEW BEST with %0.2f"%test_score)

        self.resampling.update()

        # TODO : evaluate the best genome of the whole training simulation
        # using an independent common test set, to be able to make the comparison between generations fair
        # since the fitness value between generation changes
