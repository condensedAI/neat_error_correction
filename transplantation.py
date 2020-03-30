import neat
import numpy as np

def transplantate_population(p, transplantation_file, config_rec, size_rec):
    # Load the genome to be transplantate
    if not os.path.exists(transplantation_file):
        raise ValueError("Genome file does not exist.")
    with open(transplantation_file, "rb") as f:
        genome_giv = pickle.load(f)

    # Load the board size of transplanted genome
    transplanted_dir=transplantation_file[:transplantation_file.rfind("/")]
    if not os.path.exists("%s/config.json"%transplanted_dir):
        raise ValueError("Configuration file does not exist (%s)."%("%s/config.json"%transplanted_dir))

    with open("%s/config.json"%transplanted_dir) as f:
        transplantated_config = json.load(f)

    size_giv = transplantated_config["Physics"]["distance"]

    for key, genome in p.population.items():
        # Clear genome
        genome.connections = {}
        genome.nodes = {}
        # Transplantate
        # TODO: Maybe incorporate a bit of mutation, so far the population is composed of the same individual
        transplantate(config_rec, genome, size_rec, genome_giv, size_giv)

# Transplantate the genome giver into the genome receiver
def transplantate(config_rec, genome_rec, size_rec, genome_giv, size_giv):
    if size_rec <= size_giv or (size_rec-size_giv)%2!=0:
        raise ValueError("The board size of the genomes to copy does not fit the requirements.")

    # Look-up table for the nodes in genome_giv
    lookup_keys = {}

    # Create node genes for the output pins.
    for node_key in config_rec.output_keys:
        new_node = genome_rec.create_node(config_rec, node_key)
        new_node.bias = genome_giv.nodes[node_key].bias
        genome_rec.nodes[node_key] = new_node

        lookup_keys[node_key] = node_key

    ## Add hidden nodes from to_copy_genome
    # TODO: treat case where hidden nodes are requested in config_rec
    hidden_nodes = [k for k in genome_giv.nodes if k not in config_rec.output_keys]

    for previous_key in hidden_nodes:
        node_key = config_rec.get_new_node_key(genome_rec.nodes)
        assert node_key not in genome_rec.nodes
        node = genome_rec.create_node(config_rec, node_key)
        genome_rec.nodes[node_key] = node

        # Keep track
        lookup_keys[previous_key] = node_key

    ## Add connections from to_copy_genome

    # First create the mask of nodes from to_copy_genome
    L=2*size_rec
    L_copy=2*size_giv
    to_copy_mask = np.zeros((L, L))

    for i in range(L):
        for j in range(L):
            if i < L_copy and j < L_copy:
                to_copy_mask[i,j] = True

    to_copy_mask = np.roll(to_copy_mask, size_rec - size_giv, axis=0)
    to_copy_mask = np.roll(to_copy_mask, size_rec - size_giv, axis=1)

    print(to_copy_mask)
    # Second filter out the star operator locations
    star_pos = [[x,y] for x in range(0, L,2) for y in range(0,L,2)]
    mask_star_operators = [not [x,y] in star_pos for x in range(L) for y in range(L)]

    to_copy_mask = to_copy_mask.flatten()[mask_star_operators]
    print(to_copy_mask)

    # Loop over the input nodes and finish the lookup table
    count=-1
    for i, to_copy in enumerate(to_copy_mask):
        if to_copy:
            lookup_keys[count] = -i-1
            count -= 1

    print(lookup_keys)

    # Create connections
    for input_id, output_id in genome_giv.connections:
        #print(lookup_keys[input_id], lookup_keys[output_id])
        connection = genome_rec.create_connection(config_rec, lookup_keys[input_id], lookup_keys[output_id])
        connection.weight = genome_giv.connections[input_id, output_id].weight
        connection.enabled = genome_giv.connections[input_id, output_id].enabled
        genome_rec.connections[connection.key] = connection
