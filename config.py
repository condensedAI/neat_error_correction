
GameMode = {"TRAINING": 0,
            "EVALUATION": 1}

ErrorMode = {"PROBABILISTIC": 0, # The errors are generated to a binomial distribution
             "DETERMINISTIC": 1}# The number of generated errors is fixed by error_rate

TrainingMode = {"NORMAL" : 0, #
                "RESAMPLING": 1} # adapt the training dataset by resampling samples the NN struggle to solve

RewardMode = {"BINARY": 0, # Reward is 1 for solved and 0 otherwise
              "CURRICULUM": 1} # Harder problems solved are more positively rewarded, easier problems failed are more negatively rewarded

# Default configuration
def get_default_config():
    return {
        "Physics": {
            "distance" : 3,
        },

        "Training" : {
            "n_generations" : 100,
            "network_type": 'ffnn',
            'rotation_invariant_decoder': False,
            "error_rates" : [0.01, 0.05, 0.1, 0.15],
            "error_mode": ErrorMode["PROBABILISTIC"],
            "reward_mode": RewardMode["BINARY"],
            "training_mode": TrainingMode["NORMAL"],
            "n_games" : 100,
            "max_steps" : 1000,
            "epsilon": 0.1,
            "substrate_type": 0
        },
        "Population" : {
            "pop_size" : 50,
            "initial_connection": 'full',
            "connect_add_prob" : 0.1,
            "add_node_prob" : 0.1,
            "weight_mutate_rate": 0.5,
            "bias_mutate_rate": 0.1,
            "compatibility_disjoint_coefficient" : 1,
            "compatibility_weight_coefficient" : 2,
            "compatibility_threshold" : 6,
            "species_elitism": 2,
            "activation_mutate_rate": 0,
            "activation_options": "sigmoid"
        }
}

def from_arguments(args):
    config = get_default_config()

    key_converts={"distance":"distance",
                  "numGenerations":"n_generations",
                  "networkType": "network_type",
                  "rotationInvariantDecoder": "rotation_invariant_decoder",
                  "errorMode" : "error_mode",
                  "errorRates": "error_rates",
                  "trainingMode": "training_mode",
                  "rewardMode": "reward_mode",
                  "numPuzzles": "n_games",
                  "maxSteps": "max_steps",
                  "epsilon": "epsilon",
                  "populationSize": "pop_size",
                  "initialConnection": "initial_connection",
                  "connectAddProb" : "connect_add_prob",
                  "addNodeProb": "add_node_prob",
                  "weightMutateRate" : "weight_mutate_rate",
                  "biasMutateRate": "bias_mutate_rate",
                  "compatibilityDisjointCoefficient" : "compatibility_disjoint_coefficient",
                  "compatibilityWeightCoefficient": "compatibility_weight_coefficient",
                  "compatibilityThreshold": "compatibility_threshold",
                  "speciesElitism": "species_elitism",
                  "activationMutateRate": "activation_mutate_rate",
                  "activationOptions": "activation_options",
                  "substrateType": "substrate_type"}

    for key, value in vars(args).items():
        if not value is None:
            try:
                new_key = key_converts[key]
                config[key_to_section(new_key)][new_key] = value
            except:
                print("The key %s is not recognised for config."%str(key))
                continue

    return config

def key_to_section(key):
    if key in ["distance"]:
        return "Physics"
    if key in ["n_generations", "n_games", "max_steps",
                "epsilon", "error_rates", "error_mode",
                "training_mode", "reward_mode", "network_type",
                "rotation_invariant_decoder", "substrate_type"]:
        return "Training"
    if key in ["pop_size", "connect_add_prob", "add_node_prob",
        "weight_mutate_rate", "bias_mutate_rate", "compatibility_disjoint_coefficient",
        "compatibility_weight_coefficient", "compatibility_threshold", "initial_connection",
        "species_elitism", "activation_mutate_rate", "activation_options"]:
        return "Population"

    raise ValueError("Missing key for %s"%key)

def solve_compatibilities(config):
    default_config = get_default_config()
    for key1 in default_config.keys():
        for key2 in default_config[key1].keys():
            if key1 in config and not key2 in config[key1]:
                print("%s is set to the default value: %s"%(key2, str(default_config[key1][key2])))

            elif key1 in config:
                default_config[key1][key2] = config[key1][key2]

    return default_config


def check_config(config):
    # use assert
    if "network_type" in config["Training"] and not config["Training"]["network_type"] in ['ffnn', 'cppn']:
        raise ValueError("The type of neural network should be either ffnn or cppn not %s"%config["Training"]["network_type"])
    # TODO: do the rest

    if "initial_connection" in config["Population"] and not isinstance(config["Population"]["initial_connection"], str):
        print(config["Population"]["initial_connection"])
        # This entry can take multiple strings, so we need to concatenate them
        if len(config["Population"]["initial_connection"]) > 1:
            config["Population"]["initial_connection"] = ' '.join(config["Population"]["initial_connection"])
        else:
            config["Population"]["initial_connection"] = config["Population"]["initial_connection"][0]

    if "activation_options" in config["Population"] and not isinstance(config["Population"]["activation_options"], str):
        print(config["Population"]["activation_options"])
        # This entry can take multiple strings, so we need to concatenate them
        if len(config["Population"]["activation_options"]) > 1:
            config["Population"]["activation_options"] = ' '.join(config["Population"]["activation_options"])
        else:
            config["Population"]["activation_options"] = config["Population"]["activation_options"][0]

    return solve_compatibilities(config)
