import neat
from datetime import datetime
import pickle

import game
import visualize
from simple_feed_forward import SimpleFeedForwardNetwork

class FFNNPopulation():
    def __init__(self, config):
        self.config = config
        self.d = config["Physics"]["distance"]
        self.error_rates = config["Training"]["error_rates"]
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

    def evolve(self, savedir, n_cores=1, loading_file=None, verbose=0):
        time_id = datetime.now()

        self.game = game.ToricCodeGame(self.d, self.max_steps, self.epsilon)

        if loading_file is None:
            # Generate configuration file
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
        p.add_reporter(neat.Checkpointer(generation_interval=self.n_generations/2,
                                         time_interval_seconds=600,
                                         filename_prefix="%s/checkpoint-%s-"%(savedir, time_id.strftime('%Y-%m-%d_%H-%M-%S'))))

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
            fitness += self.game.play(net, error_rate)["fitness"]
        return fitness / self.n_games
