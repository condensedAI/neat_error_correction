from sklearn.base import BaseEstimator
import neat
from datetime import datetime
import multiprocessing

import game
import visualize

verbose=True

class NeatEstimatorFFNN(BaseEstimator):
    def __init__(self, pop_size=50,
                n_generations=100,
                n_games=1,
                board_size=3,
                error_rate=0.01,
                connect_add_prob=0.5,
                add_node_prob=0.2,
                bias_mutate_rate=0.7,
                weight_mutate_rate=0.8,
                compatibility_disjoint_coefficient=1.0,
                compatibility_weight_coefficient=0.5,
                compatibility_threshold=3.0,
                max_steps=500):
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.n_games = n_games
        self.board_size = board_size
        self.error_rate = error_rate
        self.connect_add_prob=connect_add_prob
        self.add_node_prob=add_node_prob
        self.bias_mutate_rate=bias_mutate_rate
        self.weight_mutate_rate=weight_mutate_rate
        self.compatibility_disjoint_coefficient=compatibility_disjoint_coefficient
        self.compatibility_weight_coefficient=compatibility_weight_coefficient
        self.compatibility_threshold=compatibility_threshold
        self.max_steps = max_steps

    def generate_config_file(self):
        # Change the config file according to the given parameters
        dt = datetime.today()
        unique_id="{}{}{}{}{}".format(dt.year, dt.month, dt.day, dt.hour, dt.minute)

        filename="ConfigFiles/config-{}".format(unique_id)

        with open('config-toriccode-template-ffnn') as file:
            data = file.read()

            data = data.replace("{num_inputs}", str(4*self.board_size*self.board_size))
            data = data.replace("{num_outputs}", str(2*self.board_size*self.board_size))

            # Loop over the parameters of the simulation
            for param_name, param_value in self.__dict__.items():
                # Attributes like game or with_velocities do not appear in config template
                # So no need to check
                data = data.replace("{"+param_name+"}", str(param_value))

            # Create a config file corresponding to these settings
            new_file = open(filename, "w")
            new_file.write(data)
            new_file.close()

        return filename


    # Irrelevant in our case
    def fit(self, x, y):
        self.game = game.ToricCodeGame(self.board_size, self.error_rate, self.max_steps)
        return self


    def score(self, x, y=None):
        # Load configuration.
        config_file = self.generate_config_file()

        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_file)

        # Create the population, which is the top-level object for a NEAT run.
        p = neat.Population(self.config)

        if verbose:
            # Add a stdout reporter to show progress in the terminal.
            p.add_reporter(neat.StdOutReporter(True))

            #p.add_reporter(neat.Checkpointer(5))

        stats = neat.StatisticsReporter()
        p.add_reporter(stats)

        pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), self.eval_genome)
        self.winner = p.run(pe.evaluate, self.n_generations)
        #self.winner = p.run(self.eval_genomes, self.n_generations)

        # Display the winning genome.
        print('Time to reach solution: {}'.format(len(stats.generation_statistics)))
        #print('\nBest genome:\n{!s}'.format(self.winner))

        if verbose:
            # Show output of the most fit genome against training data.
            #pass
            visualize.plot_stats(stats, ylog=False, view=True)

        self.game.close()

        return self.winner.fitness
        #return len(stats.generation_statistics)


    def eval_genomes(self, genomes, config):
        for genome_id, genome in genomes:
            genome.fitness = self.eval_genome(genome, config)
            #print("Individual {}, fitness {}".format(genome_id, genome.fitness))


    def eval_genome(self, genome, config):
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness = 0
        for i in range(self.n_games):
            #print(self.game.play(net))
            fitness += self.game.play(net)
        return fitness / self.n_games
