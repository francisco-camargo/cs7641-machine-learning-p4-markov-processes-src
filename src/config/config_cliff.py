import numpy as np
import src.helper_functions as hf

cliff = {
    'problem_name': 'cliff',
    'render_mode': 'ansi',
    'show_policy': False,

    'env': {
        'env_name': 'CliffWalking-v0',
        'num_seeds': 2,
    },

    'model_params': {
        'learner_iterable' : [
            'vi',
            ],
        'gamma_iterable' : [0.5],
        'n_iters_iterable' : [100],
    },

    'simulation': {
        'sim_n_iters' : 1,
        'max_iters' : 2_000,
    },

}