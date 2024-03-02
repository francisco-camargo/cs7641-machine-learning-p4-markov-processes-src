import numpy as np
import pandas as pd
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import gymnasium as gym

def my_log_range(low, high, half_steps=False, make_int=False):
    diff = high - low
    if half_steps:
        slope = 2
    else:
        slope = 1
    numpoints = int(slope*diff+1)
    if make_int:
        return np.logspace(low, high, numpoints).astype(int)
    else:
        return np.logspace(low, high, numpoints)

def pi_convert(pi, n_states):
    '''
    Convert policy into integer value options
    '''
    return np.array(list(map(lambda x: pi(x), range(n_states))))


import os, sys
# https://stackoverflow.com/a/45669280/9205210
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

        
def get_blackjack_pickle():
    import pickle
    import os
    file_name = 'blackjack-envP'
    path = os.path.join('data', 'raw', file_name)
    with (open(path, "rb")) as open_file:
        while True:
            try:
                P = pickle.load(open_file)
            except EOFError:
                break
    return P


def num_pi_changes(env, pi_old, pi_new):
    if pi_old is None:
        return np.nan
    n_states = env.env.observation_space.n
    pi_old_array = pi_convert(pi_old, n_states)
    pi_new_array = pi_convert(pi_new, n_states)
    diff = pi_new_array - pi_old_array
    num_diff = np.sum(diff!=0)
    return num_diff


def get_env(config_dict, size, seed):
    if config_dict['problem_name'] == 'frozen':
        desc=generate_random_map(size=size, p=config_dict['env']['p'], seed=seed)
        env=gym.make(config_dict['env']['env_name'], desc=desc, render_mode=config_dict['render_mode'], is_slippery=config_dict['env']['is_slippery'])
    elif config_dict['problem_name'] == 'taxi':
        env=gym.make(config_dict['env']['env_name'], render_mode=config_dict['render_mode'])
        # env.observation_space = gym.spaces.Discrete(100)
    elif config_dict['problem_name'] == 'cliff':
        env=gym.make(config_dict['env']['env_name'], render_mode=config_dict['render_mode'])
    state, info = env.reset(seed=seed)
    # print(env.render())
    # print(f'stat: {state}')
    return env, state


def simulation(
            problem_name,
            env,
            initial_state,
            n_iters,
            pi,
            max_iters=500,
            seed=42,
            # convert_state_obs=lambda state, done: state
            ):
    test_scores = np.full([n_iters], np.nan)
    counter_list = []
    success_count = 0
    for i in range(0, n_iters):
        if problem_name == 'frozen':
            state, info = env.reset(seed=i) # I want unique seeds per simulation run, that way the transition events are different between each simulation run
                # affects the sampling of transition probabilities, so want this one for frozen-lake
        elif problem_name == 'taxi':
            # state = initial_state
            state, info = env.reset(seed=seed)
        elif problem_name == 'cliff':
            state = initial_state
        else:
            assert False, "This Problem is not accounted for"
        done = False
        # state = convert_state_obs(state, done)
        total_reward = 0
        counter = 0
        while not done:
            # print(env.render())
            # print(state)
            if counter >= max_iters: break
            action = pi(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            # next_state = convert_state_obs(next_state, done)
            state = next_state
            total_reward = reward + total_reward
            counter += 1
            if problem_name == 'frozen':
                if reward == 1.0: success_count += 1
            if problem_name == 'taxi':
                if reward == 20: success_count += 1
            # print(counter)
        # print('final total rewards')
        # print(total_reward)
        test_scores[i] = total_reward
        counter_list.append(counter)
    env.close()
    success = success_count / n_iters
    return test_scores, counter_list, success


def run_sim(
    render_mode,
    problem_name,
    env,
    initial_state,
    pi,
    num_sim_iters,
    max_sim_iters,
    seed,
    ):
    
    if render_mode == 'human': num_sim_iters = 1
    test_scores, local_counter_list, success = simulation(problem_name=problem_name, env=env.env, initial_state=initial_state, n_iters=num_sim_iters, pi=pi, max_iters=max_sim_iters, seed=seed)
    return test_scores, local_counter_list, success

def score_by_iter(df, problem_name):
    col_keep = [
        'env_name',
        'learner',
        'n_iters',
        'num_seeds',
        'is_slippery',
        'p',
        'sim_counter_limit',
        'sim_n_iters',
        'gamma',
        'init_epsilon',
        'epsilon_decay',
        'reward',
        'avg_counter',
        'max_counter',
        'success',
        'clock_time',
        'pi_diff',
        ]
    if problem_name == 'frozen': col_keep.append('size')

    groupby_list = ['env_name', 'learner', 'gamma', 'init_epsilon', 'epsilon_decay', 'n_iters']
    if problem_name == 'frozen': groupby_list.append('size')
    
    agg_dict = {
        'num_seeds': 'first',
        'is_slippery': 'first',
        'p': 'first',
        'sim_n_iters': 'first',
        'sim_counter_limit': 'first',
        }
        
    df_a = df[col_keep].groupby(groupby_list).agg(agg_dict)
    df_b = df[col_keep].groupby(groupby_list).agg(
        reward=('reward', 'mean'), reward_std=('reward', 'std'),
        clock_time=('clock_time', 'mean'), clock_time_std=('clock_time', 'std'),
        sim_counter_max=('max_counter', 'max'),
        sim_counter=('avg_counter', 'mean'), sim_counter_std=('avg_counter', 'std'),
        success=('success', 'mean'), success_std=('success', 'std'),
        pi_diff=('pi_diff', 'mean'), pi_diff_std=('pi_diff', 'std'),
        )
    
    df2 = pd.concat([df_a, df_b], axis=1)
    num_samples = len(set(df['seed']))
    df2['reward_std_of_mean'] = df2['reward_std']/(num_samples-1)**0.5
    df2['clock_time_std_of_mean'] = df2['clock_time_std']/(num_samples-1)**0.5
    df2['sim_counter_std_of_mean'] = df2['sim_counter_std']/(num_samples-1)**0.5
    df2['success_std_of_mean'] = df2['success_std']/(num_samples-1)**0.5
    df2['pi_diff_std_of_mean'] = df2['pi_diff_std']/(num_samples-1)**0.5
    return df2