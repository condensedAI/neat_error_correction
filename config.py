
GameMode = {"TRAINING": 0,
            "EVALUATION": 1}

ErrorMode = {"PROBABILISTIC": 0, # The errors are generated to a binomial distribution
             "DETERMINISTIC": 1}# The number of generated errors is fixed by error_rate


# Default configuration
def get_default_config():
    return {
        "Physics": {
            "distance" : 3,
        },

        "Training" : {
            "n_generations" : 100,
            "error_mode": ErrorMode["PROBABILISTIC"],
            "error_rates" : [0.01, 0.05, 0.1, 0.15],
            "n_games" : 100,
            "max_steps" : 1000,
            "epsilon": 0.1
        },
        "Population" : {
            "pop_size" : 50,
            "connect_add_prob" : 0.1,
            "add_node_prob" : 0.1,
            "weight_mutate_rate": 0.5,
            "bias_mutate_rate": 0.1,
            "compatibility_disjoint_coefficient" : 1,
            "compatibility_weight_coefficient" : 2,
            "compatibility_threshold" : 6
        }
}

def from_arguments(args):
    config = get_default_config()

    key_converts={"distance":"distance",
                  "numGenerations":"n_generations",
                  "errorMode" : "error_mode",
                  "errorRates": "error_rates",
                  "numPuzzles": "n_games",
                  "maxSteps": "max_steps",
                  "epsilon": "epsilon",
                  "populationSize": "pop_size",
                  "connectAddProb" : "connect_add_prob",
                  "addNodeProb": "add_node_prob",
                  "weightMutateRate" : "weight_mutate_rate",
                  "biasMutateRate": "bias_mutate_rate",
                  "compatibilityDisjointCoefficient" : "compatibility_disjoint_coefficient",
                  "compatibilityWeightCoefficient": "compatibility_weight_coefficient",
                  "compatibilityThreshold": "compatibility_threshold"}

    for key, value in vars(args).items():
        if not value is None:
            try:
                new_key = key_converts[key]
                config[key_to_section(new_key)][new_key] = value
            except:
                continue

    return config

def key_to_section(key):
    if key in ["distance"]:
        return "Physics"
    if key in ["n_generations", "n_games", "max_steps", "epsilon", "error_rates", "error_mode"]:
        return "Training"
    if key in ["pop_size", "connect_add_prob", "add_node_prob",
        "weight_mutate_rate", "bias_mutate_rate", "compatibility_disjoint_coefficient",
        "compatibility_weight_coefficient", "compatibility_threshold"]:
        return "Population"

    raise ValueError("Missing key for %s"%key)
