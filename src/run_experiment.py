from time import perf_counter
import numpy as np
import pandas as pd

from algorithms.planner import Planner
from algorithms.rl import RL

import src.helper_functions as hf
import src.plotting.plotting as plotting
import src.read_write_data as io

theta=1e-10
def train_learner(env, learner, n_iters, gamma, init_epsilon=None, epsilon_decay=None):
    with hf.HiddenPrints(): # suppress printing from this code
        time_start = perf_counter()
        if learner == 'vi':
            V, V_track, pi = Planner(env.env.P).value_iteration(gamma=gamma, n_iters=n_iters, theta=theta)
        elif learner == 'pi':
            V, V_track, pi = Planner(env.env.P).policy_iteration(gamma=gamma, n_iters=n_iters, theta=theta) 
        elif learner == 'ql':
            Q, V, pi, Q_track, pi_track = RL(env.env).q_learning(gamma=gamma, init_epsilon=init_epsilon, epsilon_decay_ratio=epsilon_decay, n_episodes=n_iters)
        time_end = perf_counter()
        time_diff = time_end - time_start
    return V, pi, time_diff
    
    
def score_learner(problem_name, env, initial_state, pi_old, pi, num_sim_iters, max_sim_iters, render_mode, seed):
    
    num_diff = hf.num_pi_changes(env=env, pi_old=pi_old, pi_new=pi)
    
    test_scores, local_counter_list, success = hf.run_sim(
        render_mode=render_mode,
        problem_name=problem_name,
        env=env,
        initial_state=initial_state,
        pi=pi,
        num_sim_iters=num_sim_iters,
        max_sim_iters=max_sim_iters,
        seed=seed,
        )

    average_score = np.mean(test_scores)
    max_counter = np.max(local_counter_list)
    average_counter = np.mean(local_counter_list)
    
    return num_diff, average_score, max_counter, average_counter, success
    

def experiment_runner(config, static_seed):
    # Initialize value collection lists
    learner_list = []
    size_list = []
    seed_list = []
    gamma_list = []
    init_epsilon_list = []
    epsilon_decay_list = []
    n_iters_list = []
    reward_list = []
    max_counter_list = []
    average_counter_list = []
    success_list = []
    diff_list = []
    time_list = []

    for learner in config['model_params']['learner_iterable']:
        print(f'Learner: {learner}')

        try: 
            size_iterable = config['env']['size_iterable']
        except KeyError:
            size_iterable = [-1]
        for lake_size in size_iterable:
            if config['problem_name'] not in ['frozen']:
                lake_size = -1
            else:
                print(f'Lake Size: {lake_size}')
            
            num_seeds = config['env']['num_seeds']
            for seed in range(num_seeds):
                np.random.seed(static_seed)
                seed_temp = seed
                seed += static_seed
                    # set seed here to ensure that we get the same map given a learner and a size
                print(f'\tSeed: {seed}, {seed_temp+1} out of {num_seeds}')
                
                env, initial_state = hf.get_env(config_dict=config, size=lake_size, seed=seed)
                
                for gamma in config['model_params']['gamma_iterable']:
                    print(f'\t\tgamma: {gamma}')
                
                    try: 
                        init_epsilon_iterable = config['model_params']['init_epsilon_iterable']
                    except KeyError:
                        init_epsilon_iterable = [-1]
                    for init_epsilon in init_epsilon_iterable:
                        
                        try: 
                            epsilon_decay_iterable = config['model_params']['epsilon_decay_iterable']
                        except KeyError:
                            epsilon_decay_iterable = [-1]
                        for epsilon_decay in epsilon_decay_iterable:
                            # if learner not in ['ql']: epsilon_decay = -1
                            
                            pi = None
                            for n_iters in config['model_params']['n_iters_iterable']:
                                # print(f'\t\t\tn_iters: {n_iters}')
                                
                                pi_old = pi
                                    
                                V, pi, time_diff = train_learner(
                                    env,
                                    learner,
                                    n_iters,
                                    gamma,
                                    init_epsilon=init_epsilon,
                                    epsilon_decay=epsilon_decay,
                                )
                                
                                num_diff, seed_average_score, max_counter, average_counter, success = score_learner(
                                    config['problem_name'],
                                    env,
                                    initial_state,
                                    pi_old,
                                    pi,
                                    num_sim_iters=config['simulation']['sim_n_iters'],
                                    max_sim_iters=config['simulation']['max_iters'],
                                    render_mode=config['render_mode'],
                                    seed=seed,
                                )
                                
                                # Summary Stats
                                learner_list.append(learner)
                                size_list.append(lake_size)
                                seed_list.append(seed)
                                n_iters_list.append(n_iters)
                                gamma_list.append(gamma)
                                init_epsilon_list.append(init_epsilon)
                                epsilon_decay_list.append(epsilon_decay)
                                reward_list.append(seed_average_score)
                                max_counter_list.append(max_counter)
                                average_counter_list.append(average_counter)
                                success_list.append(success)
                                time_list.append(time_diff)
                                diff_list.append(num_diff)
                                
                                # Plotting V and pi
                                if config['show_policy']:
                                    try:
                                        filename = config['problem_name'] + '_' + str(config['env']['size_iterable'][0]) + 'x' + str(config['env']['size_iterable'][0]) + '_p' + str(config['env']['p']) + '_' + config['model_params']['learner_iterable'][0] + '_niter' + str(n_iters)
                                        title = config['problem_name'] + ' ' + str(config['env']['size_iterable'][0]) + 'x' + str(config['env']['size_iterable'][0]) + ' p=' + str(config['env']['p']) + ' ' + config['model_params']['learner_iterable'][0] + ' n_iter=' + str(n_iters)
                                    except KeyError:
                                        title = config['problem_name'] + ' ' + config['model_params']['learner_iterable'][0] + ' n_iter=' + str(n_iters)
                                    fig = plotting.grid_values_heat_map2(env, V, pi, title, lake_size)
                                    io.save_fig(fig, 'frozen_V_plotting', filename)
                                
                            if learner in ['vi', 'pi']: break
                        if learner in ['vi', 'pi']: break
            if config['problem_name'] not in ['frozen']: break
    
    df = pd.DataFrame()
    df['learner'] = learner_list
    df['size'] = size_list
    df['seed'] = seed_list
    df['problem_name'] = config['problem_name']
    df['env_name'] = config['env']['env_name']
    df['num_seeds'] = config['env']['num_seeds']
    try: 
        is_slippery = config['env']['is_slippery']
    except KeyError:
        is_slippery = -1
    df['is_slippery'] = is_slippery
    try: 
        p = config['env']['p']
    except KeyError:
        p = -1
    df['p'] = p
    df['sim_counter_limit'] = config['simulation']['max_iters']
    df['n_iters'] = n_iters_list
    df['sim_n_iters'] = config['simulation']['sim_n_iters']
    df['gamma'] = gamma_list
    df['pi_diff'] = diff_list
    df['clock_time'] = time_list
    df['avg_counter'] = average_counter_list
    df['max_counter'] = max_counter_list
    df['reward'] = reward_list
    df['success'] = success_list
    df['init_epsilon'] = init_epsilon_list
    df['epsilon_decay'] = epsilon_decay_list
    
    return df