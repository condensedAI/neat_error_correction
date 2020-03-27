import neat
from datetime import datetime
import pickle
import os
import json

from game import ToricCodeGame
from genome_checkpointer import GenomeCheckpointer
from config import *
import visualize
from simple_feed_forward import SimpleFeedForwardNetwork
from custom_parallel_evaluator import CustomParallelEvaluator
from transplantation import transplantate_population


class FFNNPopulation():
    def __init__(self, config):
        self.config = config
        self.d = config["Physics"]["distance"]
        self.error_rates = config["Training"]["error_rates"]
        self.error_mode = config["Training"]["error_mode"]
        self.training_mode = config["Training"]["training_mode"]
        self.reward_mode = config["Training"]["reward_mode"]
        self.n_generations = config["Training"]["n_generations"]
        self.n_games = config["Training"]["n_games"]
        self.max_steps = config["Training"]["max_steps"]
        self.epsilon = config["Training"]["epsilon"]

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

    def evolve(self, savedir, n_cores=1, loading_file=None, transplantation_file=None, verbose=0):
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

            # Transplantation of genomes in the initial population
            if not transplantation_file is None:
                # Load the genome to be transplantate
                if not os.path.exists(transplantation_file):
                    raise ValueError("Genome file does not exist.")
                with open(transplantation_file, "rb") as f:
                    transplantated_genome = pickle.load(f)

                # Load the board size of transplanted genome
                transplanted_dir=transplantation_file[:transplantation_file.rfind("/")]
                if not os.path.exists("%s/config.json"%transplanted_dir):
                    raise ValueError("Configuration file does not exist (%s)."%("%s/config.json"%savedir))

                with open("%s/config.json"%transplanted_dir) as f:
                    transplantated_config = json.load(f)

                transplantate_population(p=p,
                              config_rec=population_config.genome_config,
                              size_rec=self.config["Physics"]["distance"],
                              genome_giv=transplantated_genome,
                              size_giv=transplantated_config["Physics"]["distance"])

        else:
            p = neat.Checkpointer.restore_checkpoint(loading_file)

        if verbose:
            # Add a stdout reporter to show progress in the terminal.
            p.add_reporter(neat.StdOutReporter(True))

        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        p.add_reporter(neat.Checkpointer(generation_interval=100,
                                         filename_prefix="%s/checkpoint-%s-"%(savedir, time_id.strftime('%Y-%m-%d_%H-%M-%S'))))
        p.add_reporter(GenomeCheckpointer(generation_interval=100,
                                         filename_prefix="%s/checkpoint-best-genome-%s-"%(savedir, time_id.strftime('%Y-%m-%d_%H-%M-%S'))))

        if self.training_mode == TrainingMode["RESAMPLING"]:
            pe = CustomParallelEvaluator(num_workers=n_cores,
                                        eval_function=self.eval_genome_resampling,
                                        do_resampling=True,
                                        global_test_set=True,
                                        config=self.config)
        else:
            pe = CustomParallelEvaluator(num_workers=n_cores,
                                        eval_function=self.eval_genome,
                                        do_resampling=False,
                                        global_test_set=True,
                                        config=self.config)

        w = p.run(pe.evaluate, self.n_generations)
        #print("Check best test scores: %.2f vs %.2f"%(pe.test_set.evaluate(w, population_config), pe.best_genome_test_score))
        winner = pe.best_genome

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

    def eval_genome(self, genome, config):
        #net = neat.nn.FeedForwardNetwork.create(genome, config)
        net = SimpleFeedForwardNetwork.create(genome, config)
        fitness = 0
        for i in range(self.n_games):
            # Create puzzles of varying difficulties
            error_rate = self.error_rates[i%len(self.error_rates)]
            fitness += self.game.play(net, error_rate, self.error_mode, self.reward_mode, GameMode["TRAINING"])["fitness"]
        return fitness / self.n_games

    def eval_genome_resampling(self, genome, config, puzzles_proportions):
        net = SimpleFeedForwardNetwork.create(genome, config)
        fitness = {error_rate: 0 for error_rate in self.error_rates}
        n_puzzles = {error_rate: int(self.n_games*puzzles_proportions[error_rate]) for error_rate in self.error_rates}
        fail_count = {error_rate: 0 for error_rate in self.error_rates}

        for error_rate in self.error_rates:
            for i in range(n_puzzles[error_rate]):
                result = self.game.play(net, error_rate, self.error_mode, self.reward_mode, GameMode["TRAINING"])["fitness"]
                fitness[error_rate] += result
                # Count the number of fails for the resampling learner
                fail_count[error_rate] += 1 - result

        return sum(fitness.values()) / len(self.error_rates) / self.n_games, fail_count
