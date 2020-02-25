import estimator
import estimator_without_velocities
from sklearn.model_selection import GridSearchCV
from sklearn.utils.estimator_checks import check_estimator
from numpy import save
from datetime import datetime
import pickle

#print(check_estimator(estimator.NeatEstimator))
# Grid search is not working so far

param_grid={'pop_size':[500],
            'n_generations':[200],
            'n_games':[10],
            'connect_add_prob':[0.1, 0.5],
            "add_node_prob":[0.1, 0.5],
            "weight_mutate_rate":[0.1, 0.5],
            "bias_mutate_rate":[0.1, 0.5],
            "compatibility_disjoint_coefficient" :[1.0],
            "compatibility_weight_coefficient" : [2.0],
            "compatibility_threshold" : [5]
            }

'''
param_grid={'pop_size':[10, 20],
            'n_generations':[100],
            'n_games':[1],
            'with_velocities':[False],
            'connect_add_prob':[0.1],
            "add_node_prob":[0.1]
            }
'''

search = GridSearchCV(estimator_without_velocities.NeatEstimatorWithoutVelocities(),
                      param_grid,
                      n_jobs=1)

dummy=list(range(10))
search.fit(dummy, dummy)
print(search.best_params_)

# Change the config file according to the given parameters
dt = datetime.today()
unique_id="{}{}{}{}{}".format(dt.year, dt.month, dt.day, dt.hour, dt.minute)

filename="output/results-{}.pkl".format(unique_id)

print(search.cv_results_)

f = open(filename,"wb")
pickle.dump(search.cv_results_,f)
f.close()
