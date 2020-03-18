"""
Runs evaluation functions in parallel subprocesses
in order to evaluate multiple genomes at once.
"""
from multiprocessing import Pool, Lock, Value
from multiprocessing.sharedctypes import Value
from curriculum_learner import CurriculumLearner

# The existence of this class lies on the necessity
# to share the fail_counts dictionary for curriculum learning

class ParallelEvaluatorCurriculum(object):
    def __init__(self, num_workers, eval_function, error_rates, pop_size, n_games, timeout=None):
        """
        eval_function should take one argument, a tuple of
        (genome object, config object), and return
        a single float (the genome's fitness).
        """
        self.num_workers = num_workers
        self.eval_function = eval_function
        self.timeout = timeout

        self.fail_count = {}
        self.pool = Pool(num_workers)

        self.curriculum_learner = CurriculumLearner(error_rates, pop_size, n_games, self.fail_count)
        #print("parallel init", id(self.fail_count))

    def __del__(self):
        self.pool.close() # should this be terminate?
        self.pool.join()

    def evaluate(self, genomes, config):
        self.curriculum_learner.start_generation()

        jobs = []
        for ignored_genome_id, genome in genomes:
            jobs.append(self.pool.apply_async(self.eval_function, (genome, config, self.curriculum_learner.puzzles_proportions)))
            
        # assign the fitness back to each genome
        for job, (ignored_genome_id, genome) in zip(jobs, genomes):
            genome.fitness, detailed_results = job.get(timeout=self.timeout)
            for error_rate in self.curriculum_learner.error_rates:
                self.fail_count[error_rate] += detailed_results[error_rate]

        self.curriculum_learner.end_generation()
