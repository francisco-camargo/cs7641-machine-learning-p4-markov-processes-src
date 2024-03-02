import numpy as np

# Taxi
    # https://github.com/openai/gym/blob/master/gym/envs/toy_text/blackjack.py
    # https://www.gymlibrary.dev/environments/toy_text/blackjack/
    # https://github.com/jlm429/bettermdptools/blob/master/examples/blackjack.py

    # P object
        # https://github.com/jlm429/bettermdptools/commit/f78a1c4148d2ca51d04a4cfdded7dda76b391a6a#diff-81e3d46d7343e1b5cb4b94e9641612ab1be7c9b598a05a963359bda7a5ad8717
env_name = 'Blackjack-v1'
render_mode = 'ansi' # None, 'human'
sab = True

# Value Iteration
gamma=0.1
theta=1e-10

#
n_iters_iterable = np.logspace(1, 3, 5).astype(int)
# n_iters_iterable = range(10, 70, 5)

# Simulation
num_seeds = 2
sim_n_iters = 10
