'''
Frozen Lake
    https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py
    https://www.gymlibrary.dev/environments/toy_text/frozen_lake/
    https://medium.com/analytics-vidhya/solving-the-frozenlake-environment-from-openai-gym-using-value-iteration-5a078dffe438
    https://aleksandarhaber.com/installation-and-getting-started-with-openai-gym-and-frozen-lake-environment-reinforcement-learning-tutorial/
    https://github.com/adodd202/GT-ML-Assignment4/blob/main/Frozen%20Lake%20Analysis.ipynb
'''

import numpy as np
import src.helper_functions as hf

# Optimal
frozen_vi_optimal = {
    'problem_name': 'frozen',
    'render_mode': 'ansi',
    'show_policy': False,

    'env': {
        'env_name': 'FrozenLake-v1',
        'num_seeds': 1,
        'is_slippery': True,
        'p': 0.9, # probability of state being solid ice
        'size_iterable': [8],
    },

    'model_params': {
        'learner_iterable' : [
            'vi',
            ],
        'gamma_iterable' : [0.7],
        'n_iters_iterable' : [100],
    },

    'simulation': {
        'sim_n_iters' : 20,
        'max_iters' : 2_000,
    },

}


frozen_pi_optimal = {
    'problem_name': 'frozen',
    'render_mode': 'ansi',
    'show_policy': False,

    'env': {
        'env_name': 'FrozenLake-v1',
        'num_seeds': 1,
        'is_slippery': True,
        'p': 0.9,
        'size_iterable': [8],
    },

    'model_params': {
        'learner_iterable' : [
            'pi',
            ],
        'gamma_iterable' : [0.7],
        'n_iters_iterable' : [100],
    },

    'simulation': {
        'sim_n_iters' : 20,
        'max_iters' : 2_000,
    },

}


frozen_ql_optimal = {
    'problem_name': 'frozen',
    'render_mode': 'ansi',
    'show_policy': False,

    'env': {
        'env_name': 'FrozenLake-v1',
        'num_seeds': 1,
        'is_slippery': True,
        'p': 0.9,
        'size_iterable': [8],
    },

    'model_params': {
        'learner_iterable' : [
            'ql',
            ],
        'gamma_iterable' : [0.7],
        'init_epsilon_iterable' : [0.01],
        'epsilon_decay_iterable' : [1],
        'n_iters_iterable' : [10_000]
    },

    'simulation': {
        'sim_n_iters' : 20,
        'max_iters' : 2_000,
    },

}


# V Map plotting
frozen_vi_V_plotting = {
    'problem_name': 'frozen',
    'render_mode': 'ansi',
    'show_policy': True,

    'env': {
        'env_name': 'FrozenLake-v1',
        'num_seeds': 1,
        'is_slippery': True,
        'p': 0.8,
        'size_iterable': [8],
    },

    'model_params': {
        'learner_iterable' : [
            'vi',
            ],
        'gamma_iterable' : [0.9],
        'n_iters_iterable' : [2, 3, 4, 10, 100],
    },

    'simulation': {
        'sim_n_iters' : 20,
        'max_iters' : 2_000,
    },

}


frozen_pi_V_plotting = {
    'problem_name': 'frozen',
    'render_mode': 'ansi',
    'show_policy': True,

    'env': {
        'env_name': 'FrozenLake-v1',
        'num_seeds': 1,
        'is_slippery': True,
        'p': 0.8,
        'size_iterable': [8],
    },

    'model_params': {
        'learner_iterable' : [
            'pi',
            ],
        'gamma_iterable' : [0.9],
        'n_iters_iterable' : [1, 2, 3, 4, 10, 100],
    },

    'simulation': {
        'sim_n_iters' : 20,
        'max_iters' : 2_000,
    },

}


frozen_ql_V_plotting = {
    'problem_name': 'frozen',
    'render_mode': 'ansi',
    'show_policy': True,

    'env': {
        'env_name': 'FrozenLake-v1',
        'num_seeds': 1,
        'is_slippery': True,
        'p': 0.8,
        'size_iterable': [8],
    },

    'model_params': {
        'learner_iterable' : [
            'ql',
            ],
        'gamma_iterable' : [0.9],
        'init_epsilon_iterable' : [0],
        'epsilon_decay_iterable' : [1],
        'n_iters_iterable' : [4, 10, 100, 1_000, 10_000]
    },

    'simulation': {
        'sim_n_iters' : 20,
        'max_iters' : 2_000,
    },

}


# Gamma Search
frozen_vi_gamma_search = {
    'problem_name': 'frozen',
    'render_mode': 'ansi',
    'show_policy': False,

    'env': {
        'env_name': 'FrozenLake-v1',
        'num_seeds': 20,
        'is_slippery': True,
        'p': 0.9,
        'size_iterable': [16],
    },

    'model_params': {
        'learner_iterable' : [
            'vi',
            ],
        'gamma_iterable' : [0.1, 0.3, 0.5, 0.7, 0.9],
        'n_iters_iterable' : np.logspace(0.5, 2, 10).astype(int), # hf.my_log_range(0.5, 2, True)
    },

    'simulation': {
        'sim_n_iters' : 20,
        'max_iters' : 5_000,
    },

}


frozen_pi_gamma_search = {
    'problem_name': 'frozen',
    'render_mode': 'ansi',
    'show_policy': False,

    'env': {
        'env_name': 'FrozenLake-v1',
        'num_seeds': 10,
        'is_slippery': True,
        'p': 0.9,
        'size_iterable': [16],
    },

    'model_params': {
        'learner_iterable' : [
            'pi',
            ],
        'gamma_iterable' : [0.1, 0.3, 0.5, 0.7, 0.9],
        'n_iters_iterable' : np.logspace(0, 2, 10).astype(int), # hf.my_log_range(0, 2, True) # np.logspace(1, 4, 5).astype(int)
    },

    'simulation': {
        'sim_n_iters' : 20,
        'max_iters' : 2_000,
    },

}


frozen_ql_gamma_search = {
    'problem_name': 'frozen',
    'render_mode': 'ansi',
    'show_policy': False,

    'env': {
        'env_name': 'FrozenLake-v1',
        'num_seeds': 8,
        'is_slippery': True,
        'p': 0.9, # probability of state being solid ice
        'size_iterable': [16],
    },

    'model_params': {
        'learner_iterable' : [
            'ql',
            ],
        'gamma_iterable' : [0.1, 0.3, 0.5, 0.7, 0.9], # keep ql at 0.1+, 0.7
        'init_epsilon_iterable' : [0],
        'epsilon_decay_iterable' : [1],
        'n_iters_iterable' : np.logspace(1.5, 4, 10).astype(int), # hf.my_log_range(1, 4, True)
    },

    'simulation': {
        'sim_n_iters' : 20,
        'max_iters' : 2_000,
    },

}

# epsilon search
frozen_ql_epsilon_search = {
    'problem_name': 'frozen',
    'render_mode': 'ansi',
    'show_policy': False,

    'env': {
        'env_name': 'FrozenLake-v1',
        'num_seeds': 8,
        'is_slippery': True,
        'p': 0.9, # probability of state being solid ice
        'size_iterable': [16],
    },

    'model_params': {
        'learner_iterable' : [
            'ql',
            ],
        'gamma_iterable' : [0.7],
        'init_epsilon_iterable' : [0, 0.1, 0.3, 0.5, 1], # np.logspace(-1, 0, 4), # hf.my_log_range(-1, 0, True), # [0.003162],
        'epsilon_decay_iterable' : [1],
        'n_iters_iterable' : np.logspace(1, 4, 10).astype(int), # hf.my_log_range(1, 4, True)
    },

    'simulation': {
        'sim_n_iters' : 20,
        'max_iters' : 2_000,
    },

}

# gridsearch
frozen_ql_gridsearch_search = {
    'problem_name': 'frozen',
    'render_mode': 'ansi',
    'show_policy': False,

    'env': {
        'env_name': 'FrozenLake-v1',
        'num_seeds': 10,
        'is_slippery': True,
        'p': 0.8, # probability of state being solid ice
        'size_iterable': [8],
    },

    'model_params': {
        'learner_iterable' : [
            'ql',
            ],
        'gamma_iterable' : [0.7], # keep ql at 0.1+, 0.7
        'init_epsilon_iterable' : hf.my_log_range(-2, 0, True), # [0.003162],
        'epsilon_decay_iterable' : hf.my_log_range(0, 0, True),
        'n_iters_iterable' : hf.my_log_range(3, 3, True, make_int=True), # np.logspace(1, 4, 5).astype(int)
    },

    'simulation': {
        'sim_n_iters' : 20,
        'max_iters' : 2_000,
    },

}


# Vary Size
frozen_vary_size_vi = {
    'problem_name': 'frozen',
    'render_mode': 'ansi',
    'show_policy': False,

    'env': {
        'env_name': 'FrozenLake-v1',
        'num_seeds': 10,
        'is_slippery': True,
        'p': 0.9,
        'size_iterable': [4, 8, 12, 16],
    },

    'model_params': {
        'learner_iterable' : [
            'vi',
            ],
        'gamma_iterable' : [0.7],
        'n_iters_iterable' : np.logspace(0.5, 2, 10).astype(int), # [3, 6, 10, 32, 100]
    },

    'simulation': {
        'sim_n_iters' : 20,
        'max_iters' : 10_000,
    },
}

frozen_vary_size_pi = {
    'problem_name': 'frozen',
    'render_mode': 'ansi',
    'show_policy': False,

    'env': {
        'env_name': 'FrozenLake-v1',
        'num_seeds': 10,
        'is_slippery': True,
        'p': 0.9,
        'size_iterable': [4, 8, 12, 16],
    },

    'model_params': {
        'learner_iterable' : [
            'pi',
            ],
        'gamma_iterable' : [0.7],
        'n_iters_iterable' : np.logspace(0, 1.5, 10).astype(int), # [1, 3, 10, 32]
    },

    'simulation': {
        'sim_n_iters' : 20,
        'max_iters' : 10_000,
    },
}

frozen_vary_size_ql = {
    'problem_name': 'frozen',
    'render_mode': 'ansi',
    'show_policy': False,

    'env': {
        'env_name': 'FrozenLake-v1',
        'num_seeds': 10,
        'is_slippery': True,
        'p': 0.9,
        'size_iterable': [4, 8, 12, 16],
    },

    'model_params': {
        'learner_iterable' : [
            'ql',
            ],
        'gamma_iterable' : [0.7],
        'init_epsilon_iterable' : [0.01],
        'epsilon_decay_iterable' : [1],
        'n_iters_iterable' : np.logspace(1, 4, 10).astype(int), # [10, 31, 100, 312, 1_000, 10_000]
    },

    'simulation': {
        'sim_n_iters' : 20,
        'max_iters' : 10_000,
    },
}