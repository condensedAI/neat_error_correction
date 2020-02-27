from neat_ffnn import NeatFFNN
from datetime import datetime
import argparse, json
import os


def simulate(config, savedir, n_jobs, verbose):
    time_id = datetime.now()

    # In case the directory already exist and a configuration file also
    # We just load it
    if savedir is not None and os.path.exists(savedir):
        with open("%s/config.json"%savedir, "r") as f:
            config = json.load(f)

    # Otherwise we create it
    # Along with a directory regardless of whether a directory has been given
    else:
        if savedir is None:
            savedir = "output/default-%s"%time_id.strftime('%Y-%m-%d_%H-%M-%S')

        os.mkdir(savedir)
        # Save the configuration dict
        with open("%s/config.json"%savedir, 'w') as f:
            json.dump(config, f, indent=4)


    game = NeatFFNN(config)

    results = game.run(savedir, n_jobs, verbose)

    elapsed = datetime.now() - time_id
    print("Total running time:", elapsed.seconds,":",elapsed.microseconds)

    return results

if __name__ == "__main__":
    # Parse arguments passed to the program (or set defaults)
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", "--saveDir", help="Config file to load (overrides other settings)")
    #parser.add_argument("-id", "--id", help="Identifier for different runs for instance")
    parser.add_argument("--numParallelJobs", type=int, default=1, help="Number of jobs launched in parallel")
    parser.add_argument("-v", "--verbose", type=int, choices=[0,1,2], default=0, help="Level of verbose output (higher is more)")
    parser.add_argument("-L", "--distance", type=int, choices=[3,5,7], default=3, help="Toric Code Distance")
    parser.add_argument("--errorRate", type=float, default=0.1, help="Qubit error rate")
    parser.add_argument("--numGenerations", type=int, default=100, help="Number of simulated generations")
    parser.add_argument("--numPuzzles", type=int, default=100, help="Number of syndrome configurations to solve per individual")
    parser.add_argument("--maxSteps", type=int, default=1000, help="Number of maximum qubits flips to solve syndromes")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Epsilon of the greedy search among perspectives results")
    parser.add_argument("--populationSize", type=int, default=50, help="Size of the population")
    parser.add_argument("--connectAddProb", type=float, default=0.1, help="Probability of adding a new connection")
    parser.add_argument("--addNodeProb", type=float, default=0.1, help="Probability of adding a new node")
    parser.add_argument("--weightMutateRate", type=float, default=0.5, help="Mutation rate for the weights")
    parser.add_argument("--biasMutateRate", type=float, default=0.1, help="Mutation rate for the bias")
    parser.add_argument("--compatibilityDisjointCoefficient", type=float, default=1.0, help="Weight on the number of disjoint genes used when calculating genome distance")
    parser.add_argument("--compatibilityWeightCoefficient", type=float, default=2.0, help="Weight on the L2 distance between connections weights used when calculating genome distance")
    parser.add_argument("--compatibilityThreshold", type=float, default=6.0, help="Distance threshold to form species")
    args = parser.parse_args()

    config = {
        "Physics": {
            "distance" : args.distance,
            "error_rate" : args.errorRate, # Should be removed at some point
        }

        "Training" : {
            "n_generations" : args.numGenerations,
            "n_games" : args.numPuzzles,
            "max_steps" : args.maxSteps,
            "epsilon": args.epsilon
        },
        "Population" : {
            "pop_size" : args.populationSize,
            "connect_add_prob" : args.connectAddProb,
            "add_node_prob" : args.addNodeProb,
            "weight_mutate_rate": args.weightMutateRate,
            "bias_mutate_rate": args.biasMutateRate,
            "compatibility_disjoint_coefficient" : args.compatibilityDisjointCoefficient,
            "compatibility_weight_coefficient" : args.compatibilityWeightCoefficient,
            "compatibility_threshold" : args.compatibilityThreshold
        }
    }

    simulate(config, args.saveDir, args.numParallelJobs, num.verbose)
