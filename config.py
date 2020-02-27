

# Default configuration
def get_default_config():
    return {
        "Physics": {
            "distance" : 3,
            "error_rate" : 0.1, # Should be removed at some point
        },

        "Training" : {
            "n_generations" : 100,
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

def key_to_section(key):
    if key in ["distance", "error_rate"]:
        return "Physics"
    if key in ["n_generations", "n_games", "max_steps", "epsilon"]:
        return "Training"
    if key in ["pop_size", "connect_add_prob", "add_node_prob",
        "weight_mutate_rate", "bias_mutate_rate", "compatibility_disjoint_coefficient",
        "compatibility_weight_coefficient", "compatibility_threshold"]:
        return "Population"

    raise ValueError("Missing key for %s"%key)
