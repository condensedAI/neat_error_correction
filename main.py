import estimator_ffnn
import datetime

board_size=3
error_rate=0.1

pop_size=50
n_generations=200
n_games=100
connect_add_prob=0.1
add_node_prob=0.1
weight_mutate_rate =0.1
bias_mutate_rate = 0.1
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient = 2.0
compatibility_threshold = 5

start = datetime.datetime.now()

game = estimator_ffnn.NeatEstimatorFFNN(pop_size,
                                   n_generations,
                                   n_games,
                                   board_size,
                                   error_rate,
                                   connect_add_prob,
                                   add_node_prob,
                                   weight_mutate_rate,
                                   bias_mutate_rate,
                                   compatibility_disjoint_coefficient,
                                   compatibility_weight_coefficient,
                                   compatibility_threshold)

game.fit(None, None)
game.score(None)


elapsed = datetime.datetime.now() - start
print(elapsed.seconds,":",elapsed.microseconds)
