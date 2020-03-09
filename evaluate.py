import pickle
from datetime import datetime
import argparse, json
import os
import pandas as pd
import numpy as np

import neat

from game import *

from simple_feed_forward import SimpleFeedForwardNetwork


def evaluate(file, error_rates, n_games, n_jobs, verbose):
    time_id = datetime.now()

    np.random.seed(0)

    # Load the corresponding config files
    savedir = file[:file.rfind("/")]

    if not os.path.exists("%s/config.json"%savedir):
        raise ValueError("Configuration file does not exist (%s)."%("%s/config.json"%savedir))

    with open("%s/config.json"%savedir) as f:
        config = json.load(f)

    # Create a game
    game = ToricCodeGame(board_size=config["Physics"]["distance"],
                           max_steps=config["Training"]["max_steps"],
                           epsilon=0)

    # Load the genome to be evaluated
    if not os.path.exists(file):
        raise ValueError("Genome file does not exist.")

    with open(file, "rb") as f:
        genome = pickle.load(f)

    if not os.path.exists("%s/population-config"%savedir):
        raise ValueError("Population configuration file does not exist.")

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         "%s/population-config"%savedir)

    #net = neat.nn.FeedForwardNetwork.create(genome, config)
    net = SimpleFeedForwardNetwork.create(genome, config)

    fitness = []
    # /TODO\ Parallelize this part
    results={"fitness":[], "error_rate":[], "outcome":[], "nsteps":[]}
    for error_rate in error_rates:
        fitness.append(0)
        for i in range(n_games):
            result = game.play(net, error_rate, GameMode.EVALUATION)
            fitness[-1] += result["fitness"]
            #print("Result", i, result)

            for k, v in result.items():
                results[k].append(v)

        fitness[-1] /= n_games
        print("Evaluation on error_rate=%.2f is done."%error_rate)

    elapsed = datetime.now() - time_id
    print("Total running time:", elapsed.seconds,":",elapsed.microseconds)

    # Always overwrite the result of evaluation
    # Synthesis report
    savefile = "%s_evaluation.ngames=%i.csv"%(file.replace(".pkl", ""), n_games)
    if os.path.exists(savefile):
        print("Deleting evaluation file %s"%savefile)
        os.remove(savefile)

    print([error_rates, fitness])
    df = pd.DataFrame(list(zip(error_rates, fitness)), columns=["error_rate", "mean_fitness"])
    df.to_csv(savefile)

    # Detailed report
    savefile = "%s_detailed_results_evaluation.ngames=%i.csv"%(file.replace(".pkl", ""), n_games)
    if os.path.exists(savefile):
        print("Deleting evaluation file %s"%savefile)
        os.remove(savefile)

    pd.DataFrame.from_dict(results).to_csv(savefile)


    return error_rates, fitness

if __name__ == "__main__":
    # Parse arguments passed to the program (or set defaults)
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", nargs="+", help="Genome file to load and evaluate")
    parser.add_argument("--errorRates", type=float, nargs="+", default=[0.01, 0.05, 0.1, 0.15], help="Qubit error rate")
    parser.add_argument("--numPuzzles", type=int, default=100, help="Number of syndrome configurations to solve per individual")
    #parser.add_argument("--maxSteps", type=int, default=1000, help="Number of maximum qubits flips to solve syndromes")
    parser.add_argument("-j", "--numParallelJobs", type=int, default=1, help="Number of jobs launched in parallel")
    parser.add_argument("-v", "--verbose", type=int, choices=[0,1,2], default=0, help="Level of verbose output (higher is more)")
    args = parser.parse_args()

    for file in args.file:
        evaluate(file, args.errorRates, args.numPuzzles, args.numParallelJobs, args.verbose)
