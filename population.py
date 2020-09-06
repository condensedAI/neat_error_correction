import neat
from datetime import datetime
import pickle
import os
import json
import visualize

from parallel_evaluator import ParallelEvaluator
from parallel_evaluator_resampling import ParallelEvaluatorResampling
from transplantation import transplantate_population
from genome_checkpointer import GenomeCheckpointer
from config import *


class Population():
    def __init__(self, config):
        self.config = config
        self.d = config["Physics"]["distance"]
        self.training_mode = config["Training"]["training_mode"]
        self.n_generations = config["Training"]["n_generations"]
        self.network_type = config["Training"]["network_type"]
        self.substrate_type = config["Training"]["substrate_type"]

    def generate_config_file(self, savedir):
        # Change the config file according to the given parameters
        with open('config-toric-code-template-%s'%(self.network_type)) as file:
            data = file.read()

            if self.network_type == "ffnn":
                data = data.replace("{num_inputs}", str(3*self.d*self.d))
            if self.network_type == "cppn":
                if self.substrate_type == 0:
                    data = data.replace("{num_inputs}", str(4))
                if self.substrate_type == 1:
                    data = data.replace("{num_inputs}", str(2))

            # No {num_outputs} occurence in config file for cppn
            if self.config["Training"]["rotation_invariant_decoder"]:
                data = data.replace("{num_outputs}", str(1))
            else:
                data = data.replace("{num_outputs}", str(4))

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
            # TODO: adapt to hyperNEAT
            if self.network_type == "ffnn" and not transplantation_file is None:
                transplantate_population(p=p,
                              transplantation_file=transplantation_file,
                              config_rec=population_config.genome_config,
                              size_rec=self.config["Physics"]["distance"])

        else:
            p = neat.Checkpointer.restore_checkpoint(loading_file)

        if verbose:
            # Add a stdout reporter to show progress in the terminal.
            p.add_reporter(neat.StdOutReporter(True))

        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        p.add_reporter(neat.Checkpointer(generation_interval=100,
                                         time_interval_seconds=None,
                                         filename_prefix="%s/checkpoint-%s-"%(savedir, time_id.strftime('%Y-%m-%d_%H-%M-%S'))))
        p.add_reporter(GenomeCheckpointer(generation_interval=100,
                                         filename_prefix="%s/checkpoint-best-genome-%s-"%(savedir, time_id.strftime('%Y-%m-%d_%H-%M-%S'))))
        # TODO: checkpointer is cleaner for reporting best genome performance
        #p.add_reporter(Test)
                                         
        if self.training_mode == TrainingMode["RESAMPLING"]:
            pe = ParallelEvaluatorResampling(num_workers=n_cores,
                                             global_test_set=True,
                                             config=self.config,
                                             savedir=savedir)
        else:
            pe = ParallelEvaluator(num_workers=n_cores,
                                   global_test_set=True,
                                   config=self.config,
                                   savedir=savedir)

        w = p.run(pe.evaluate, self.n_generations)
        #print("Check best test scores: %.2f vs %.2f"%(pe.test_set.evaluate(w, population_config), pe.best_genome_test_score))
        winner = pe.best_genome

        # Display the winning genome.
        print('\nBest genome on global test set:\n{!s}'.format(winner))

        if verbose > 1:
            # Show output of the most fit genome against training data.
            visualize.plot_stats(stats, ylog=False, view=True)

        # Saving and closing
        stats.save_genome_fitness(filename="%s/genome.fitness.%s.csv"%(savedir, time_id.strftime('%Y-%m-%d_%H-%M-%S')))
        stats.save_species_count(filename="%s/species.count.%s.csv"%(savedir, time_id.strftime('%Y-%m-%d_%H-%M-%S')))
        stats.save_species_fitness(filename="%s/species.fitness.%s.csv"%(savedir, time_id.strftime('%Y-%m-%d_%H-%M-%S')))

        # Save the winner
        with open("%s/winner.genome.%s.pkl"%(savedir, time_id.strftime('%Y-%m-%d_%H-%M-%S')), 'wb') as f:
            pickle.dump(winner, f)

        # HyperNEAT: save also the substrate
        if (self.network_type == 'cppn'):
            # TODO
            pass


        return winner.fitness
