from numpy import mean, std
from datetime import datetime
import pickle
import itertools
import argparse, os

import train
import config as cf

config = cf.get_default_config()

# Number of cross-validation
n_runs = 2

# Grid-search parameters
param_grid={'pop_size':[200],
            'n_generations':[100],
            'n_games':[100],
            #'error_rate':[0.01, 0.05, 0.1, 0.15],
            'error_rate':[0.05, 0.1, 0.15],
            'distance':[5],
            'connect_add_prob':[0.1],
            "add_node_prob":[0.1],
            "weight_mutate_rate":[0.5],
            "bias_mutate_rate":[0.1],
            "compatibility_disjoint_coefficient" :[1.0],
            "compatibility_weight_coefficient" : [2.0],
            "compatibility_threshold" : [5]
            }

# Create the parameter to loop over
keys=[]
parameters=[]
for k, v in param_grid.items():
    if len(v)==1:
        config[cf.key_to_section(k)][k] = v[0]
    else:
        keys+=[k]
        parameters+=[v]

parameters = list(itertools.product(*parameters))

parser = argparse.ArgumentParser()
parser.add_argument("-dir", "--saveDir", help="Config file to load (overrides other settings)")
parser.add_argument("--numParallelJobs", type=int, default=1, help="Number of jobs launched in parallel")
parser.add_argument("-v", "--verbose", type=int, choices=[0,1,2], default=0, help="Level of verbose output (higher is more)")
args = parser.parse_args()

if args.saveDir is None or os.path.exists(args.saveDir):
    raise ValueError("Need a directory path or one that is not already existing")
else:
    rootdir = "output/grid-search-%s"%args.saveDir
    os.mkdir(rootdir)

# Initialize the results dictionary
results={"mean_fitness":[], "std_fitness":[], "config":[]}
for n in range(n_runs):
    results["run%i_fitness"%n] = []
for key in keys:
    results["param_%s"%key] = []

# Launch the grid search
for i, param in enumerate(parameters):
    # Create configuration file
    for n, v in enumerate(list(param)):
        config[cf.key_to_section(keys[n])][keys[n]] = v
        results["param_"+keys[n]].append(v)

    results["config"].append(config)

    savedir = "%s/set%i"%(rootdir, i)
    set_results=[]
    for n in range(n_runs):
        result = train.simulate(config, savedir, args.numParallelJobs, args.verbose)
        results["run%i_fitness"%n].append(result)
        set_results.append(result)

    # Aggregate the results
    results["mean_fitness"].append(mean(set_results))
    results["std_fitness"].append(std(set_results))

print(results)
filename="%s/results.pkl"%rootdir
with open(filename,"wb") as f:
    pickle.dump(results,f)
