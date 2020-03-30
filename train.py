from ffnn_population import FFNNPopulation
from datetime import datetime
import argparse, json
from glob import glob
import os
from numpy import argmax

from config import *

def simulate(config, savedir, n_jobs, loading_mode, transplantation_file, verbose):
    time_id = datetime.now()

    if not savedir is None:
        savedir = "output/%s"%savedir.replace("output/", "")
    ckpt_file = None

    # In case the directory already exist and a configuration file also
    # We just load it
    if savedir is not None and os.path.exists(savedir):
        with open("%s/config.json"%savedir, "r") as f:
            config = json.load(f)

        # TODO: Allow for modifications of config when loading (like training longer, ect..)

        # Verify checkpoints exist
        if loading_mode:
            if len(glob("%s/checkpoint*"%savedir)) == 0:
                raise ValueError("No checkpoint to load.")
            else:
                ckpts=glob("%s/checkpoint*"%savedir)
                ckpt_file=ckpts[argmax(map(os.path.getmtime, ckpts))]
                print("Loading the last edited checkpoint %s"%ckpt_file)

    # Otherwise we create it
    # Along with a directory regardless of whether a directory has been given
    else:
        if savedir is None:
            savedir = "output/default-%s"%time_id.strftime('%Y-%m-%d_%H-%M-%S')

        os.mkdir(savedir)
        # Save the configuration dict
        with open("%s/config.json"%savedir, 'w') as f:
            json.dump(config, f, indent=4)


    population = FFNNPopulation(config)

    results = population.evolve(savedir, n_jobs, ckpt_file, transplantation_file, verbose)

    elapsed = datetime.now() - time_id
    print("Total running time:", elapsed.seconds,":",elapsed.microseconds)

    return results

if __name__ == "__main__":
    # Parse arguments passed to the program (or set defaults)
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", "--saveDir", help="Config file to load (overrides other settings)")
    #parser.add_argument("-id", "--id", help="Identifier for different runs for instance")
    parser.add_argument("-j", "--numParallelJobs", type=int, default=1, help="Number of jobs launched in parallel")
    parser.add_argument("-v", "--verbose", type=int, choices=[0,1,2], default=0, help="Level of verbose output (higher is more)")
    parser.add_argument("--load", default=False, action="store_true", help="Loading an already existing population")
    parser.add_argument('--transplantate', help="Genome file to transplantate the population")

    parser.add_argument("-L", "--distance", type=int, choices=[3,5,7], help="Toric Code Distance")
    parser.add_argument("--errorRates", type=float, nargs="+", help="Qubit error rate")
    parser.add_argument("--errorMode", type=int, choices=[0,1], help="Error generation mode")
    parser.add_argument("--trainingMode", type=int, choices=[0,1], help="Training mode")
    parser.add_argument("--rewardMode", type=int, choices=[0,1], help="Reward mode")
    parser.add_argument("--numGenerations", type=int, help="Number of simulated generations")
    parser.add_argument("--initialConnection", help="Initial connection of the initial NN in the population")	    
    parser.add_argument("--numPuzzles", type=int, help="Number of syndrome configurations to solve per individual")
    parser.add_argument("--maxSteps", type=int, help="Number of maximum qubits flips to solve syndromes")
    parser.add_argument("--epsilon", type=float, help="Epsilon of the greedy search among perspectives results")
    parser.add_argument("--populationSize", type=int, help="Size of the population")
    parser.add_argument("--connectAddProb", type=float, help="Probability of adding a new connection")
    parser.add_argument("--addNodeProb", type=float, help="Probability of adding a new node")
    parser.add_argument("--weightMutateRate", type=float, help="Mutation rate for the weights")
    parser.add_argument("--biasMutateRate", type=float, help="Mutation rate for the bias")
    parser.add_argument("--compatibilityDisjointCoefficient", type=float, help="Weight on the number of disjoint genes used when calculating genome distance")
    parser.add_argument("--compatibilityWeightCoefficient", type=float, help="Weight on the L2 distance between connections weights used when calculating genome distance")
    parser.add_argument("--compatibilityThreshold", type=float, help="Distance threshold to form species")
    args = parser.parse_args()

    config = from_arguments(args)

    print(config)

    simulate(config, args.saveDir, args.numParallelJobs, args.load, args.transplantate, args.verbose)
