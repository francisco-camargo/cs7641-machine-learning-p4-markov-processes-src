# Taxi
    # https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py
    # https://www.gymlibrary.dev/environments/toy_text/taxi/
    # https://www.kaggle.com/code/angps95/intro-to-reinforcement-learning-with-openai-gym
    # https://www.youtube.com/watch?v=1i0MnGILhec&t=597s
    # https://www.kaggle.com/code/angps95/intro-to-reinforcement-learning-with-openai-gym/notebook#3.-Policy-Iteration/Value-Iteration

import numpy as np
import src.helper_functions as hf

taxi_optimal_vi = {
    'problem_name': 'taxi',
    'render_mode': 'ansi',
    'show_policy': False,

    'env': {
        'env_name': 'Taxi-v3',
        'num_seeds': 20,
    },

    'model_params': {
        'learner_iterable' : [
            'ql',
            ],
        'gamma_iterable' : [0.99],
        'n_iters_iterable' : [100],
    },

    'simulation': {
        'sim_n_iters' : 1, # not stochastic, so no need to run multiple times
        'max_iters' : 50,
    },
}

taxi_optimal_pi = {
    'problem_name': 'taxi',
    'render_mode': 'ansi',
    'show_policy': False,

    'env': {
        'env_name': 'Taxi-v3',
        'num_seeds': 20,
    },

    'model_params': {
        'learner_iterable' : [
            'pi',
            ],
        'gamma_iterable' : [0.7], # want some speed
        'n_iters_iterable' : [100],
    },

    'simulation': {
        'sim_n_iters' : 1, # not stochastic, so no need to run multiple times
        'max_iters' : 50,
    },
}

taxi_optimal_ql = {
    'problem_name': 'taxi',
    'render_mode': 'ansi',
    'show_policy': False,

    'env': {
        'env_name': 'Taxi-v3',
        'num_seeds': 20,
    },

    'model_params': {
        'learner_iterable' : [
            'ql',
            ],
        'gamma_iterable' : [0.999], # gamma search -> 0.999, previous gridsearch -> 0.99
        'init_epsilon_iterable' : [0.999],
        'epsilon_decay_iterable' : [0.9],
        'n_iters_iterable' : [10_000],
    },

    'simulation': {
        'sim_n_iters' : 1, # not stochastic, so no need to run multiple times
        'max_iters' : 50,
    },
}

# V plotting
taxi_vi_V_plotting = {
    'problem_name': 'taxi',
    'render_mode': 'ansi',
    'show_policy': True,

    'env': {
        'env_name': 'Taxi-v3',
        'num_seeds': 1,
    },

    'model_params': {
        'learner_iterable' : [
            'vi',
            ],
        'gamma_iterable' : [0.5],
        'n_iters_iterable' : [20],
    },

    'simulation': {
        'sim_n_iters' : 1, # not stochastic, so no need to run multiple times
        'max_iters' : 50,
    },
}

# Gamma search
taxi_vi_gamma_search = {
    'problem_name': 'taxi',
    'render_mode': 'ansi',
    'show_policy': False,

    'env': {
        'env_name': 'Taxi-v3',
        'num_seeds': 50,
    },

    'model_params': {
        'learner_iterable' : [
            'vi',
            ],
        'gamma_iterable' : [0.1, 0.3, 0.5, 0.7, 0.9],
        'n_iters_iterable' : np.logspace(0.5, 2, 20).astype(int), # hf.my_log_range(0.5, 2, True, make_int=True),
    },

    'simulation': {
        'sim_n_iters' : 1, # not stochastic, so no need to run multiple times
        'max_iters' : 50,
    },
}

taxi_pi_gamma_search = {
    'problem_name': 'taxi',
    'render_mode': 'ansi',
    'show_policy': False,

    'env': {
        'env_name': 'Taxi-v3',
        'num_seeds': 50,
    },

    'model_params': {
        'learner_iterable' : [
            'pi',
            ],
        'gamma_iterable' : [0.5],
        'n_iters_iterable' : [14,15,16], #np.logspace(0.5, 2, 20).astype(int),# hf.my_log_range(0.5, 2, True, make_int=True),
    },

    'simulation': {
        'sim_n_iters' : 1, # not stochastic, so no need to run multiple times
        'max_iters' : 50,
    },
}

taxi_ql_gamma_search = {
    'problem_name': 'taxi',
    'render_mode': 'ansi',
    'show_policy': False,

    'env': {
        'env_name': 'Taxi-v3',
        'num_seeds': 10,
    },

    'model_params': {
        'learner_iterable' : [
            'ql',
            ],
        'gamma_iterable' : [0.1, 0.3, 0.5, 0.7, 0.9, 0.999],
        'init_epsilon_iterable' : [0.999],
        'epsilon_decay_iterable' : [0.9],
        'n_iters_iterable' : np.logspace(2, 4, 10).astype(int) # hf.my_log_range(2, 4.5, True, make_int=True),
    },

    'simulation': {
        'sim_n_iters' : 1, # not stochastic, so no need to run multiple times
        'max_iters' : 50,
    },
}

# epsilon search
taxi_ql_epsilon_search = {
    'problem_name': 'taxi',
    'render_mode': 'ansi',
    'show_policy': False,

    'env': {
        'env_name': 'Taxi-v3',
        'num_seeds': 10,
    },

    'model_params': {
        'learner_iterable' : [
            'ql',
            ],
        'gamma_iterable' : [0.999],
        'init_epsilon_iterable' : [0, 0.1, 0.5, 0.9, 1], # hf.my_log_range(-3, 0, False),
        'epsilon_decay_iterable' : [0.9],
        'n_iters_iterable' : np.logspace(2, 4, 10).astype(int) # hf.my_log_range(2, 4.5, True, make_int=True),
    },

    'simulation': {
        'sim_n_iters' : 1, # not stochastic, so no need to run multiple times
        'max_iters' : 50,
    },
}

# epsilon decay search
taxi_ql_epsilon_decay_search = {
    'problem_name': 'taxi',
    'render_mode': 'ansi',
    'show_policy': False,

    'env': {
        'env_name': 'Taxi-v3',
        'num_seeds': 20,
    },

    'model_params': {
        'learner_iterable' : [
            'ql',
            ],
        'gamma_iterable' : [1],
        'init_epsilon_iterable' : [1], # hf.my_log_range(-3, 0, False),
        'epsilon_decay_iterable' : [0.1, 0.5, 0.9, 1],
        'n_iters_iterable' : np.logspace(2, 4, 10).astype(int) # hf.my_log_range(2, 4.5, True, make_int=True),
    },

    'simulation': {
        'sim_n_iters' : 1, # not stochastic, so no need to run multiple times
        'max_iters' : 50,
    },
}

taxi_ql_gridsearch_search = {
    'problem_name': 'taxi',
    'render_mode': 'ansi',
    'show_policy': False,

    'env': {
        'env_name': 'Taxi-v3',
        'num_seeds': 20,
    },

    'model_params': {
        'learner_iterable' : [
            'ql',
            ],
        'gamma_iterable' : [1],
        'init_epsilon_iterable' : [1], # hf.my_log_range(-2, -2, True),
        'epsilon_decay_iterable' : [0.1, 0.3, 0.5, 0.7, 0.9, 1], # hf.my_log_range(0, 0, True),
        'n_iters_iterable' : [2000], # hf.my_log_range(3.5, 3.5, True, make_int=True),
    },

    'simulation': {
        'sim_n_iters' : 1, # not stochastic, so no need to run multiple times
        'max_iters' : 50,
    },
}