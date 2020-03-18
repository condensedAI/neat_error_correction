import neat
from datetime import datetime
import pickle

from game import ToricCodeGame
from genome_checkpointer import GenomeCheckpointer
from config import *
import visualize
from simple_feed_forward import SimpleFeedForwardNetwork
from curriculum_learner import CurriculumLearner
from parallel_evaluator_curriculum import ParallelEvaluatorCurriculum

class FFNNPopulation():
    def __init__(self, config):
        self.config = config
        self.d = config["Physics"]["distance"]
        self.error_rates = config["Training"]["error_rates"]
        self.error_mode = config["Training"]["error_mode"]
        self.training_mode = config["Training"]["training_mode"]
        self.n_generations = config["Training"]["n_generations"]
        self.n_games = config["Training"]["n_games"]
        self.max_steps = config["Training"]["max_steps"]
        self.epsilon = config["Training"]["epsilon"]

        # For the curriculum learning
        #if self.training_mode == TrainingMode["CURRICULUM"]:
        #    self.curriculum_learner = CurriculumLearner(self.error_rates, config["Population"]["pop_size"], self.n_games)

    def generate_config_file(self, savedir):
        # Change the config file according to the given parameters
        with open('config-toric-code-template-ffnn') as file:
            data = file.read()

            #data = data.replace("{num_inputs}", str(4*self.d*self.d))
            data = data.replace("{num_inputs}", str(3*self.d*self.d))

            # Loop over the parameters of the simulation
            for param_name, param_value in self.config["Population"].items():
                # Attributes like n_games or epsilon do not appear in config template
                # So no need to check
                data = data.replace("{"+param_name+"}", str(param_value))

            # Create a config file corresponding to these settings
            new_file = open(savedir+"/population-config", "w")
            new_file.write(data)
            new_file.close()

    def evolve(self, savedir, n_cores=1, loading_file=None, verbose=0):
        time_id = datetime.now()

        self.game = ToricCodeGame(self.d, self.max_steps, self.epsilon)

        if loading_file is None:
            # Generate configuration file
            # TODO: No need to generate population config file each time
            self.generate_config_file(savedir)

            population_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                 savedir+"/population-config")

            # Create the population, which is the top-level object for a NEAT run.
            p = neat.Population(population_config)
        else:
            p = neat.Checkpointer.restore_checkpoint(loading_file)

        if verbose:
            # Add a stdout reporter to show progress in the terminal.
            p.add_reporter(neat.StdOutReporter(True))

            #p.add_reporter(neat.Checkpointer(5))

        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        p.add_reporter(neat.Checkpointer(generation_interval=100,
                                         filename_prefix="%s/checkpoint-%s-"%(savedir, time_id.strftime('%Y-%m-%d_%H-%M-%S'))))
        p.add_reporter(GenomeCheckpointer(generation_interval=100,
                                         filename_prefix="%s/checkpoint-best-genome-%s-"%(savedir, time_id.strftime('%Y-%m-%d_%H-%M-%S'))))

        if self.training_mode == TrainingMode["CURRICULUM"]:
            #p.add_reporter(self.curriculum_learner)
            pe = ParallelEvaluatorCurriculum(n_cores, self.eval_genome_curriculum_learning, self.error_rates, self.config["Population"]["pop_size"], self.n_games)
        else:
            pe = neat.ParallelEvaluator(n_cores, self.eval_genome)

        winner = p.run(pe.evaluate, self.n_generations)

        # Display the winning genome.
        print('\nBest genome:\n{!s}'.format(winner))

        if verbose >1:
            # Show output of the most fit genome against training data.
            #pass
            visualize.plot_stats(stats, ylog=False, view=True)

        # Saving and closing
        stats.save_genome_fitness(filename="%s/genome.fitness.%s.csv"%(savedir, time_id.strftime('%Y-%m-%d_%H-%M-%S')))
        stats.save_species_count(filename="%s/species.count.%s.csv"%(savedir, time_id.strftime('%Y-%m-%d_%H-%M-%S')))
        stats.save_species_fitness(filename="%s/species.fitness.%s.csv"%(savedir, time_id.strftime('%Y-%m-%d_%H-%M-%S')))

        # Save the winner
        with open("%s/winner.genome.%s.pkl"%(savedir, time_id.strftime('%Y-%m-%d_%H-%M-%S')), 'wb') as f:
            pickle.dump(winner, f)

        self.game.close()

        return winner.fitness
        #return len(stats.generation_statistics)

    def eval_genome(self, genome, config):
        #net = neat.nn.FeedForwardNetwork.create(genome, config)
        net = SimpleFeedForwardNetwork.create(genome, config)
        fitness = 0
        for i in range(self.n_games):
            # Create puzzles of varying difficulties
            error_rate = self.error_rates[i%len(self.error_rates)]
            fitness += self.game.play(net, error_rate, self.error_mode, GameMode["TRAINING"], 0)["fitness"]
        return fitness / self.n_games

    def eval_genome_curriculum_learning(self, genome, config, puzzles_proportions):
        net = SimpleFeedForwardNetwork.create(genome, config)
        fitness = {error_rate: 0 for error_rate in self.error_rates}
        n_puzzles = {error_rate: int(self.n_games*puzzles_proportions[error_rate]) for error_rate in self.error_rates}
        fail_count = {error_rate: 0 for error_rate in self.error_rates}
        for error_rate in self.error_rates:
            for i in range(n_puzzles[error_rate]):
                result = self.game.play(net, error_rate, self.error_mode, GameMode["TRAINING"])["fitness"]
                fitness[error_rate] += result
                # Count the number of fails for the curriculum learner
                #print(id(fail_count))
                fail_count[error_rate] += 1 - result

            #fitness[error_rate] /= n_puzzles[error_rate]

        #return sum(fitness.values()) / len(self.error_rates) , fail_count
        return sum(fitness.values()) / len(self.error_rates) / self.n_games, fail_count
