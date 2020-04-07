from neat.graphs import feed_forward_layers
from collections import namedtuple
from neat.genes import DefaultConnectionGene
import numpy as np
import copy
from scipy.special import expit

''' EFFICIENT IMPLEMENTATION
This type of neural network has
    - only one type of activation function per layer
    - aggregation_function is set to summing
which allows for matrix multiplication and speed up of the activation computation
'''

Layer = namedtuple("Layer", ["input_global_keys", "output_global_keys", \
                             "weight_matrix", "bias_vector", \
                             "act_function"])

# Vectorized version of activation function
# /!\ Actually the sigmoid from neat-python is sigmoid(5*x)
# The factor 5 is integrated in the weights and bias
def sigmoid(x):
    return expit(x)

class PhenotypeNetwork(object):
    def __init__(self, layers, global_output_keys, total_nodes):
        ''' param: layers of the neural network
        '''

        self.layers = layers
        self.global_output_keys = global_output_keys
        self.total_nodes = total_nodes

    def activate(self, input_values):
        if len(self.layers) == 0:
            return [0]*len(self.global_output_keys)
        if len(self.layers[0].input_global_keys) > len(input_values):
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.layers[0].input_global_keys), len(input_values)))

        # Create a big vector which contains the value of the nodes
        node_values = np.zeros(self.total_nodes)
        # /!\ Be careful maybe not all the input values are taken
        node_values[:len(input_values)] = input_values

        for l in self.layers:
            # Compute the vector of values of the next layer
            values = l.weight_matrix.dot(node_values[l.input_global_keys]) + l.bias_vector

            node_values[l.output_global_keys] = l.act_function(values)

        return node_values[self.global_output_keys]

    @staticmethod
    def query_cppn_weight(coord1, coord2, cppn, coord_diff=False, max_weight=5.0):
        if not coord_diff:
            i = [coord1[0], coord1[1], coord2[0], coord2[1]]
        else:
            i = [coord1[0]-coord2[0], coord1[1]-coord2[1]]

        w = cppn.activate(i)[0]

        if abs(w) > 0.2:  # If abs(weight) is below threshold, treat weight as 0.0.
            return w * max_weight
        else:
            return 0.0

    @staticmethod
    def query_cppn_bias(coord2, cppn, coord_diff=False):
        if not coord_diff:
            i = [0, 0, coord2[0], coord2[1]]
        else:
            i = [coord2[0], coord2[1]]
        return cppn.activate(i)[1]

    @staticmethod
    def create(cppn_network, substrate):
        """ Receives a CPPN genome and returns its NN phenotype (a FeedForwardNetwork). """

        # Create FFNN genome from CPPN phenotype
        # TODO: Maybe not optimal
        input_keys = [-i -1 for i in range(len(substrate.input_coordinates))]
        output_keys = [i for i in range(len(substrate.output_coordinates))]
        num_outputs = len(output_keys)

        # Input - Output connections
        genome_connections = {}
        for ikey, ipos in zip(input_keys, substrate.input_coordinates):
            for okey, opos in zip(output_keys, substrate.output_coordinates):
                weight = PhenotypeNetwork.query_cppn_weight(ipos, opos, cppn_network, substrate.with_coord_diff)
                if weight != 0.0:
                    genome_connections[(ikey, okey)] = DefaultConnectionGene((ikey, okey))
                    genome_connections[(ikey, okey)].enabled = True
                    genome_connections[(ikey, okey)].weight = weight

        # Output nodes bias
        genome_nodes_bias = {okey : 0 for okey in output_keys}
        for okey, opos in zip(output_keys, substrate.output_coordinates):
            genome_nodes_bias[okey] = PhenotypeNetwork.query_cppn_bias(opos, cppn_network, substrate.with_coord_diff)

        # Gather expressed connections.
        connections = [cg.key for cg in genome_connections.values() if cg.enabled]

        computational_layers = feed_forward_layers(input_keys, output_keys, connections)

        # Insert the input layer as the first layer
        # It matters that the input layer is inserted as a list and not a set, otherwise ordering'd be messed up
        computational_layers.insert(0, input_keys)
        visited_nodes = copy.copy(input_keys)

        # Each entry of the dict stores the global id of a node
        global_ids={}
        global_count=0
        global_OUT_keys=np.zeros(num_outputs, dtype=int)
        for i, l in enumerate(computational_layers):
            for n in l:
                global_ids[n] = global_count

                # If the node is one the output of the NN, keep memory
                if n in output_keys:
                    global_OUT_keys[n] = global_count

                global_count+=1

        build_layers=[]

        # Start with the first hidden "layer"
        for l in computational_layers[1:]:
            input_global_keys=[]
            output_global_keys=[]
            weight_matrix = []
            bias_vector = []

            #print(l)

            # Go over the nodes in the computational layer
            for no, okey in enumerate(l):
                output_global_keys.append(global_ids[okey])
                bias_vector.append(genome_nodes_bias[okey])

                # Fill in the weight matrix
                for ikey in visited_nodes:

                    if (ikey, okey) in connections:
                        # If we created already this row in the weight matrix
                        if global_ids[ikey] in input_global_keys:
                            row_index = input_global_keys.index(global_ids[ikey])
                            weight_matrix[row_index][no] = genome_connections[(ikey, okey)].weight
                        else:

                            input_global_keys.append(global_ids[ikey])
                            weight_matrix.append(np.zeros(len(l)))
                            weight_matrix[-1][no] = genome_connections[(ikey, okey)].weight


                        connections.remove((ikey, okey))

            visited_nodes += list(l)

            # Only the sigmoid activation function is allowed
            act_function = sigmoid
            # Conversion
            input_global_keys=np.array(input_global_keys)
            output_global_keys=np.array(output_global_keys)
            # The factor 5 is here to match the definition of sigmoid in neat-python library
            weight_matrix=5*np.matrix(weight_matrix).transpose()
            bias_vector=5*np.array(bias_vector)
            '''
            else:
                act_function = config.genome_config.activation_defs.get(ng.activation)
                input_global_keys=np.array(input_global_keys)
                output_global_keys=np.array(output_global_keys)
                weight_matrix=np.matrix(weight_matrix).transpose()
                bias_vector=np.array(bias_vector)
            '''

            build_layers.append(Layer(input_global_keys, output_global_keys, weight_matrix, bias_vector, act_function))

        return PhenotypeNetwork(build_layers, global_OUT_keys, global_count)
