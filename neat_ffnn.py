import neat
from datetime import datetime
import pickle

import game
import visualize

class NeatFFNN():
    def __init__(self, config):
        self.config = config
        self.d = config["distance"]
        self.error_rate = config["error_rate"]
        self.n_generations = config["Training"]["n_generations"]
        self.n_games = config["Training"]["n_games"]
        self.max_steps = config["Training"]["max_steps"]
        self.epsilon = config["Training"]["epsilon"]

    def generate_config_file(self, savedir):
        # Change the config file according to the given parameters
        with open('config-toric-code-template-ffnn') as file:
            data = file.read()

            data = data.replace("{num_inputs}", str(4*self.d*self.d))

            # Loop over the parameters of the simulation
            for param_name, param_value in self.config["Population"].items():
                # Attributes like game or with_velocities do not appear in config template
                # So no need to check
                data = data.replace("{"+param_name+"}", str(param_value))

            # Create a config file corresponding to these settings
            new_file = open(savedir+"/population-config", "w")
            new_file.write(data)
            new_file.close()


    def run(self, savedir, n_cores=1, verbose=0):
        time_id = datetime.now()

        self.game = game.ToricCodeGame(self.d, self.error_rate, self.max_steps, self.epsilon)

        # Load configuration.
        self.generate_config_file(savedir)

        population_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             savedir+"/population-config")

        # Create the population, which is the top-level object for a NEAT run.
        p = neat.Population(population_config)

        if verbose:
            # Add a stdout reporter to show progress in the terminal.
            p.add_reporter(neat.StdOutReporter(True))

            #p.add_reporter(neat.Checkpointer(5))

        stats = neat.StatisticsReporter()
        p.add_reporter(stats)

        pe = neat.ParallelEvaluator(n_cores, self.eval_genome)
        winner = p.run(pe.evaluate, self.n_generations)

        # Display the winning genome.
        print('Time to reach solution: {}'.format(len(stats.generation_statistics)))
        print('\nBest genome:\n{!s}'.format(winner))

        if verbose:
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
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness = 0
        for i in range(self.n_games):
            #print(self.game.play(net))
            fitness += self.game.play(net)
        return fitness / self.n_games