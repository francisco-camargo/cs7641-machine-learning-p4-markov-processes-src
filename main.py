# https://github.com/hakantekgul/cs7641-assignment4/blob/d0d4093152ea0dc6319e27ca38375fa18b55b564/assignment_4.py

import matplotlib.pyplot as plt

import src.helper_functions as hf
import src.run_experiment as run_experiment

import src.read_write_data as io
import src. plotting.plotting as plotting

calculate = True
plot_make = True
plot_show = False

experiment_list = [
    # 'frozen_vi_optimal',
    # 'frozen_pi_optimal',
    # 'frozen_ql_optimal',
    
    # 'frozen_vi_V_plotting',
    # 'frozen_pi_V_plotting',
    # 'frozen_ql_V_plotting',
    
    'frozen_vi_gamma_search',
    # 'frozen_pi_gamma_search',
    # 'frozen_ql_gamma_search',
    
    # 'frozen_ql_epsilon_search',
    
    # 'frozen_ql_gridsearch_search',
    
    # 'frozen_vary_size_vi',
    # 'frozen_vary_size_pi',
    # 'frozen_vary_size_ql',
    
    
    # Taxi
    # 'taxi_vi_V_plotting',
    
    # 'taxi_vi_gamma_search',
    # 'taxi_pi_gamma_search',
    # 'taxi_ql_gamma_search',
    
    # 'taxi_ql_gridsearch_search',
    
    # 'taxi_ql_epsilon_decay_search',
    # 'taxi_ql_epsilon_search',
    
    # Cliff
    # 'cliff',
]

def get_config(experiment):

    # Frozen Lake Experiments
    if   experiment == 'frozen_vi_optimal': from src.config.frozen_experiment_configs import frozen_vi_optimal as config_dict
    elif experiment == 'frozen_pi_optimal': from src.config.frozen_experiment_configs import frozen_pi_optimal as config_dict
    elif experiment == 'frozen_ql_optimal': from src.config.frozen_experiment_configs import frozen_ql_optimal as config_dict
    
    elif experiment == 'frozen_vi_V_plotting': from src.config.frozen_experiment_configs import frozen_vi_V_plotting as config_dict
    elif experiment == 'frozen_pi_V_plotting': from src.config.frozen_experiment_configs import frozen_pi_V_plotting as config_dict
    elif experiment == 'frozen_ql_V_plotting': from src.config.frozen_experiment_configs import frozen_ql_V_plotting as config_dict
    
    elif experiment == 'frozen_vi_gamma_search': from src.config.frozen_experiment_configs import frozen_vi_gamma_search as config_dict
    elif experiment == 'frozen_pi_gamma_search': from src.config.frozen_experiment_configs import frozen_pi_gamma_search as config_dict
    elif experiment == 'frozen_ql_gamma_search': from src.config.frozen_experiment_configs import frozen_ql_gamma_search as config_dict
    
    elif experiment == 'frozen_ql_epsilon_search': from src.config.frozen_experiment_configs import frozen_ql_epsilon_search as config_dict

    elif experiment == 'frozen_ql_gridsearch_search': from src.config.frozen_experiment_configs import frozen_ql_gridsearch_search as config_dict
    
    elif experiment == 'frozen_vary_size_vi': from src.config.frozen_experiment_configs import frozen_vary_size_vi as config_dict
    elif experiment == 'frozen_vary_size_pi': from src.config.frozen_experiment_configs import frozen_vary_size_pi as config_dict
    elif experiment == 'frozen_vary_size_ql': from src.config.frozen_experiment_configs import frozen_vary_size_ql as config_dict
    
    
    # Taxi Experiments
    elif experiment == 'taxi_vi_V_plotting': from src.config.config_taxi import taxi_vi_V_plotting as config_dict
    elif experiment == 'taxi_pi_V_plotting': from src.config.config_taxi import taxi_pi_V_plotting as config_dict
    elif experiment == 'taxi_ql_V_plotting': from src.config.config_taxi import taxi_ql_V_plotting as config_dict
    
    elif experiment == 'taxi_vi_gamma_search': from src.config.config_taxi import taxi_vi_gamma_search as config_dict
    elif experiment == 'taxi_pi_gamma_search': from src.config.config_taxi import taxi_pi_gamma_search as config_dict
    elif experiment == 'taxi_ql_gamma_search': from src.config.config_taxi import taxi_ql_gamma_search as config_dict
    
    elif experiment == 'taxi_ql_epsilon_search': from src.config.config_taxi import taxi_ql_epsilon_search as config_dict
    elif experiment == 'taxi_ql_epsilon_decay_search': from src.config.config_taxi import taxi_ql_epsilon_decay_search as config_dict
    
    elif experiment == 'taxi_ql_gridsearch_search': from src.config.config_taxi import taxi_ql_gridsearch_search as config_dict

    
    # Cliff
    elif experiment == 'cliff': from src.config.config_cliff import cliff as config_dict
    
    return config_dict

if calculate:
    for experiment in experiment_list:
        print(experiment)
        config_dict = get_config(experiment)

        static_seed = 42
        df = run_experiment.experiment_runner(config=config_dict, static_seed=static_seed)

        df2 = hf.score_by_iter(df, problem_name=config_dict['problem_name'])

        io.save_df_data(
            df=df,
            data_folder='interim',
            experiment_name=experiment,
            filename=f'raw_{experiment}.csv'
            )
        io.save_df_data(
            df=df2,
            data_folder='interim',
            experiment_name=experiment,
            filename=f'{experiment}.csv'
            )


if plot_make:
    
    for experiment in experiment_list:
        
        config_dict = get_config(experiment)
    
        df = io.read_df_data(
            data_folder='interim',
            experiment_name=experiment,
            filename=f'raw_{experiment}',
        )
        
        df2 = io.read_df_data(
            data_folder='interim',
            experiment_name=experiment,
            filename=experiment,
        )
        
        print()
        print(df)
        print()
        # print(df2[['size', 'clock_time', 'pi_diff', 'reward', 'success', 'success_std', 'success_std_of_mean', 'sim_counter', 'sim_counter_max']])
        print(df2[['n_iters', 'gamma', 'clock_time', 'pi_diff', 'reward', 'success', 'success_std', 'success_std_of_mean', 'sim_counter', 'sim_counter_max']])
        
        if experiment in ['frozen_vi_gamma_search', 'frozen_pi_gamma_search', 'frozen_ql_gamma_search',
                          'taxi_vi_gamma_search', 'taxi_pi_gamma_search', 'taxi_ql_gamma_search']:
            plotting.gamma_search_plotter(df2, config_dict, experiment)
            
        if experiment in ['frozen_ql_epsilon_search',
                          'taxi_ql_epsilon_search']:
            plotting.epsilon_search_plotter(df2, config_dict, experiment)
            
        if experiment in ['frozen_ql_epsilon_decay_search',
                          'taxi_ql_epsilon_decay_search']:
            plotting.epsilon_decay_search_plotter(df2, config_dict, experiment)
            
        if experiment in ['frozen_vary_size_vi', 'frozen_vary_size_pi', 'frozen_vary_size_ql']:
            plotting.vary_size_plotter(df2, config_dict, experiment)
        
    if plot_show: plt.show()
