import pickle
from datetime import datetime
import argparse, json
import os
import pandas as pd
import numpy as np
from multiprocessing import Pool
import multiprocessing

import neat

from config import GameMode, RewardMode, check_config
from neat.nn import FeedForwardNetwork
from game import ToricCodeGame
from simple_feed_forward import SimpleFeedForwardNetwork
from phenotype_network import PhenotypeNetwork
from substrates import *




def evaluate(file, error_rates, error_mode, n_games, n_jobs, verbose, file_suffix='', transfer_to_distance=None):
    time_id = datetime.now()

    # Load the corresponding config files
    savedir = file[:file.rfind("/")]

    if not os.path.exists("%s/config.json"%savedir):
        raise ValueError("Configuration file does not exist (%s)."%("%s/config.json"%savedir))

    with open("%s/config.json"%savedir) as f:
        config = json.load(f)

    config = check_config(config)

    # Load the genome to be evaluated
    if not os.path.exists(file):
        raise ValueError("Genome file does not exist.")

    with open(file, "rb") as f:
        genome = pickle.load(f)

    if not os.path.exists("%s/population-config"%savedir):
        raise ValueError("Population configuration file does not exist.")

    population_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         "%s/population-config"%savedir)


    if config["Training"]["network_type"] == 'ffnn':
        net = SimpleFeedForwardNetwork.create(genome, population_config)
    elif config["Training"]["network_type"] == 'cppn':
        # HyperNEAT: possibility of evaluating a CPPN trained on d=3 data on d>3 data
        if transfer_to_distance is None:
            code_distance = config["Physics"]["distance"]
            connection_weight_scale = 1
        elif transfer_to_distance > config["Physics"]["distance"]:
            code_distance = transfer_to_distance
            # As they are more connections, in larger codes, we need to scale down the connection weight by this factor
            connection_weight_scale = config["Physics"]["distance"]**2 / transfer_to_distance**2
            #connection_weight_scale = 0.01
        else:
            raise ValueError("Transfer knwoledge can only be done to higher distance codes.")

        if config["Training"]["substrate_type"] == 0:
            substrate = SubstrateType0(code_distance, config["Training"]["rotation_invariant_decoder"])
        elif config["Training"]["substrate_type"] == 1:
            substrate = SubstrateType1(code_distance)

        #print(code_distance, connection_weight_scale)
        cppn_network = FeedForwardNetwork.create(genome, population_config)
        net = PhenotypeNetwork.create(cppn_network, substrate, connection_weight_scale)


    # DIRTY: To ensure that samples are generated according to transfer_to_distance
    config["Physics"]["distance"] = code_distance

    ## (PARALLEL) EVALUATION LOOP
    fitness = []
    results={"fitness":[], "error_rate":[], "outcome":[], "nsteps":[], "initial_qubits_flips":[]}

    # with statement to close properly the parallel processes
    with Pool(n_jobs) as pool:
        # Game evaluation
        for error_rate in error_rates:
            fitness.append(0)

            jobs=[]
            for i in range(n_games):
                #
                jobs.append(pool.apply_async(get_fitness, (net, config, error_rate, error_mode)))

            for job in jobs:
                output, errors_id = job.get(timeout=None)

                fitness[-1] += output["fitness"]
                for k, v in output.items():
                    results[k].append(v)
                results["initial_qubits_flips"].append(errors_id)

            fitness[-1] /= n_games
            print("Evaluation on error_rate=%.2f is done, %.2f success."%(error_rate, fitness[-1]))

        elapsed = datetime.now() - time_id
        print("Total running time:", elapsed.seconds,":",elapsed.microseconds)

        # Always overwrite the result of evaluation
        # Synthesis report
        if transfer_to_distance is not None:
            file_suffix+=".transfered_distance%i"%transfer_to_distance

        savefile = "%s_evaluation.ngames=%i.errormode=%i.%s.csv"%(file.replace(".pkl", ""), n_games, error_mode, file_suffix)
        if os.path.exists(savefile):
            print("Deleting evaluation file %s"%savefile)
            os.remove(savefile)

        print([error_rates, fitness])
        df = pd.DataFrame(list(zip(error_rates, fitness)), columns=["error_rate", "mean_fitness"])
        df.to_csv(savefile)

        # Detailed report
        savefile = "%s_detailed_results_evaluation.ngames=%i.%s.csv"%(file.replace(".pkl", ""), n_games, file_suffix)
        if os.path.exists(savefile):
            print("Deleting evaluation file %s"%savefile)
            os.remove(savefile)

        pd.DataFrame.from_dict(results).to_csv(savefile)

    return error_rates, fitness

def get_fitness(net, config, error_rate, error_mode, seed=None):
    # We need to create a different game object for each thread
    game = ToricCodeGame(config)
    res = game.play(net, error_rate, error_mode, RewardMode["BINARY"], GameMode["EVALUATION"], seed)
    initial_errors = ['1' if i in game.env.initial_qubits_flips else '0' for i in game.env.state.qubit_pos]
    error_id = int(''.join(initial_errors),2)
    return res, error_id

if __name__ == "__main__":
    # Parse arguments passed to the program (or set defaults)
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", nargs="+", help="Genome file to load and evaluate")
    parser.add_argument("--errorRates", type=float, nargs="+", default=np.arange(0.01, 0.16, 0.01), help="Qubit error rate")
    parser.add_argument("--errorMode", type=int, choices=[0,1], default=0, help="Error generation mode")
    parser.add_argument("-n", "--numPuzzles", type=int, default=1000, help="Number of syndrome configurations to solve per individual")
    #parser.add_argument("--maxSteps", type=int, default=1000, help="Number of maximum qubits flips to solve syndromes")
    parser.add_argument("-j", "--numParallelJobs", type=int, default=1, help="Number of jobs launched in parallel")
    parser.add_argument("--id", default="", help="File additional id")
    parser.add_argument("--transferToDistance", type=int, choices=[3,5,7,9,11], help="Hyperneat: Toric code distance to evaluate on")
    parser.add_argument("-v", "--verbose", type=int, choices=[0,1,2], default=0, help="Level of verbose output (higher is more)")
    args = parser.parse_args()

    for file in args.file:
        evaluate(file, args.errorRates, args.errorMode, args.numPuzzles, args.numParallelJobs, args.verbose, args.id, args.transferToDistance)
