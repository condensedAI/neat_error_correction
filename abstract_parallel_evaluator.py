"""
Runs evaluation functions in parallel subprocesses
in order to evaluate multiple genomes at once.
"""
from multiprocessing import Pool
import copy

from test_set import TestSet

#
class AbstractParallelEvaluator(object):
    def __init__(self, num_workers, config, global_test_set=True, timeout=None):
        self.num_workers = num_workers
        self.timeout = timeout
        self.pool = Pool(num_workers)

        # Keeping track of the best genome evaluated on the exact same test set
        self.global_test_set = global_test_set
        if global_test_set:
            self.best_genome = None
            self.best_genome_test_score = 0
            self.test_set = TestSet(config, n_games=4000)

    def __del__(self):
        self.pool.close() # should this be terminate?
        self.pool.join()

    def test(self, generation_best, config):
        # Evaluate the best genome of the generation on the test set
        test_score = self.test_set.evaluate(self.pool, generation_best, config)
        if test_score > self.best_genome_test_score:
            # Make sure to do a deep copy
            self.best_genome = copy.copy(generation_best)
            self.best_genome_test_score = test_score
            print("NEW BEST with %0.2f"%test_score)
