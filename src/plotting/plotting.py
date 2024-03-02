import numpy as np
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import warnings
from matplotlib.colors import LinearSegmentedColormap

try:
    import src.helper_functions as hf
except ModuleNotFoundError:
    import helper_functions as hf
    
import src.plotting.plot_end_results as plot_end_results
import src.read_write_data as io


def grid_world_policy_plot(data, label):
    if not math.modf(math.sqrt(len(data)))[0] == 0.0:
        warnings.warn("Grid map expected.  Check data length")
    else:
        try:
            data = np.around(np.array(data).reshape((8, 8)), 2)
        except ValueError:
            data = np.around(np.array(data).reshape((4, 4)), 2)
        df = pd.DataFrame(data=data)
        my_colors = ((0.0, 0.0, 0.0, 1.0), (0.8, 0.0, 0.0, 1.0), (0.0, 0.8, 0.0, 1.0), (0.0, 0.0, 0.8, 1.0))
        cmap = LinearSegmentedColormap.from_list('Custom', my_colors, len(my_colors))
        plt.figure()
        ax = sns.heatmap(df, cmap=cmap, linewidths=1.0)
        colorbar = ax.collections[0].colorbar
        colorbar.set_ticks([.4, 1.1, 1.9, 2.6])
        colorbar.set_ticklabels(['Left', 'Down', 'Right', 'Up'])
        plt.title(label)


def grid_values_heat_map(data, label):
    if not math.modf(math.sqrt(len(data)))[0] == 0.0:
        warnings.warn("Grid map expected.  Check data length")
    else:
        try:
            data = np.around(np.array(data).reshape((8, 8)), 2)
        except ValueError:
            data = np.around(np.array(data).reshape((4, 4)), 2)
        df = pd.DataFrame(data=data)
        plt.figure()
        sns.heatmap(df, annot=True).set_title(label)


def v_iters_plot(data, label):
    df = pd.DataFrame(data=data)
    df.columns = [label]
    sns.set_theme(style="whitegrid")
    title = label + " v Iterations"
    sns.lineplot(x=df.index, y=label, data=df).set_title(title)
    plt.show()
    
def heat_maps(env, V, pi):
    grid_values_heat_map(V, "State Values")
    n_states = env.env.observation_space.n
    grid_world_policy_plot(hf.pi_convert(pi, n_states), "Grid World Policy")
    plt.show()
    

fontsize = 9
fontsize_ticks = fontsize - 2
fig_dim_x = 3.2
fig_dim_y = fig_dim_x * 0.75
alpha = 0.2

from matplotlib.colors import ListedColormap
def grid_values_heat_map2(env, data, policy_values, label, size):
    '''
    env - from gym.make()
    data - V (state value data)
    policy values = [pi(state) for state in range(0,size**2)]
    label - title of the plot
    size = int  (ex. size=8 for an 8x8 space)
    '''
    policy_values = hf.pi_convert(pi=policy_values, n_states=env.env.observation_space.n)
    if not math.modf(math.sqrt(len(data)))[0] == 0.0:
        warnings.warn("Grid map expected.  Check data length")
    else:
        actions = np.array(policy_values).reshape(size,size)
        mapping = {
            0: '←',
            1: '↓',
            2: '→',
            3: '↑'
        }
        s_position = np.where(env.desc == b'S')
        s_position = list(zip(s_position[0], s_position[1]))
        g_position = np.where(env.desc == b'G')
        g_position = list(zip(g_position[0], g_position[1]))
        h_positions = np.where(env.desc == b'H')
        h_positions_list = list(zip(h_positions[0], h_positions[1]))
        mask = np.full_like(data, False, dtype=bool).reshape(size, size)
        for pos in h_positions_list:
            row,col = pos
            mask[row,col] = True
        mask2 = np.full_like(data, False, dtype=bool).reshape(size, size)
        mask2[0,0] = True
        mask2[-1,-1] = True
        
        cyan_color = sns.color_palette("bright", 10)[-1]
        data = np.around(np.array(data).reshape((size, size)), 2)
        df = pd.DataFrame(data=data)
        
        fig, ax = plt.subplots()
        fig.set_size_inches(fig_dim_x, fig_dim_y)
        
        ax2 = sns.heatmap(df, annot=False, vmin=-3, vmax=1).set_title(label, fontsize=fontsize)
        cax = ax2.figure.axes[-1]
        cax.tick_params(labelsize=fontsize_ticks)
        ax2.figure.axes[-1].set_ylabel('V', size=fontsize)
        
        sns.heatmap(df, cmap=ListedColormap(['indigo']), mask=~mask, cbar=False)
        sns.heatmap(df, cmap=ListedColormap(['black']), mask=~mask2, cbar=False)
        for i in range(size):
            for j in range(size):
                if (i, j) not in h_positions_list and (i, j) not in g_position:
                    ax.text(j + 0.5, i + 0.5, mapping[actions[i, j]], ha='center', va='center', color='cyan', size=12, weight='bold')
        for row, col in h_positions_list:
            ax.text(col + 0.5, row + 0.5, 'H', ha='center', va='center', color='white', weight='bold')
        for row, col in s_position:
            ax.text(col + 0.5, row + 0.5, 'S', ha='center', va='center', color='red', weight='bold')
        for row, col in g_position:
            ax.text(col + 0.5, row + 0.5, 'G', ha='center', va='center', color='red', weight='bold')
        plt.xticks(fontsize=fontsize_ticks)
        plt.yticks(fontsize=fontsize_ticks)
        plt.tight_layout(pad=0)
    
    return fig


def plot_vs_n_iters(experiment, config_dict, df, title, legend_loc='best'):
    # xlim = [1, list(config_dict['model_params']['n_iters_iterable'])[-1]]
    xlim = [list(config_dict['model_params']['n_iters_iterable'])[0], list(config_dict['model_params']['n_iters_iterable'])[-1]]
    plt_success = plot_end_results.generic_plotter(
                df=df,
                independent_variable='n_iters',
                dependent_variable='success',
                dependent_variable_halfband='success_std_of_mean',
                xscale='log',
                xlim=xlim,
                ylim=[-0.05, 1.05],
                title = title,
                legend_loc=legend_loc,
                show=False,
                )
    io.save_fig(plt_object=plt_success, experiment_name=experiment, filename=f'{experiment}_success')
    
    plt_reward = plot_end_results.generic_plotter(
                df=df,
                independent_variable='n_iters',
                dependent_variable='reward',
                dependent_variable_halfband='reward_std_of_mean',
                xscale='log',
                xlim=xlim,
                # ylim=[-125, 25],
                title = title,
                legend_loc=legend_loc,
                show=False,
                )
    io.save_fig(plt_object=plt_reward, experiment_name=experiment, filename=f'{experiment}_reward')
    
    plt_pi_diff = plot_end_results.generic_plotter(
                df=df,
                independent_variable='n_iters',
                dependent_variable='pi_diff',
                dependent_variable_halfband='pi_diff_std_of_mean',
                xscale='log',
                xlim=xlim,
                title = title,
                legend_loc=legend_loc,
                show=False,
                )
    io.save_fig(plt_object=plt_pi_diff, experiment_name=experiment, filename=f'{experiment}_pi_diff')
    
    plt_clock_time = plot_end_results.generic_plotter(
                df=df,
                independent_variable='n_iters',
                dependent_variable='clock_time',
                dependent_variable_halfband='clock_time_std_of_mean',
                xscale='log',
                yscale='log',
                xlim=xlim,
                title = title,
                legend_loc=legend_loc,
                show=False,
                )
    io.save_fig(plt_object=plt_clock_time, experiment_name=experiment, filename=f'{experiment}_clock_time')
    
    plt.close('all')
    
    
def gamma_search_plotter(df, config_dict, experiment):
    # Gamma search experiments
    df_gamma = df.set_index(['gamma'])
    try:
        title = config_dict['problem_name'] + ' ' + str(config_dict['env']['size_iterable'][0])+'x'+str(config_dict['env']['size_iterable'][0])+' p='+str(config_dict['env']['p'])+ ' '+config_dict['model_params']['learner_iterable'][0]
    except:
        title = config_dict['problem_name'] + ' ' + config_dict['model_params']['learner_iterable'][0]
    plot_vs_n_iters(experiment, config_dict, df=df_gamma, title=title)
    

def epsilon_search_plotter(df, config_dict, experiment):
    df_gamma = df.set_index(['init_epsilon'])
    try:
        title = config_dict['problem_name'] + ' ' + str(config_dict['env']['size_iterable'][0])+'x'+str(config_dict['env']['size_iterable'][0])+' p='+str(config_dict['env']['p'])+ ' '+config_dict['model_params']['learner_iterable'][0]
    except:
        title = config_dict['problem_name'] + ' ' + config_dict['model_params']['learner_iterable'][0]
    plot_vs_n_iters(experiment, config_dict, df=df_gamma, title=title)
    
def epsilon_decay_search_plotter(df, config_dict, experiment):
    df_gamma = df.set_index(['epsilon_decay'])
    try:
        title = config_dict['problem_name'] + ' ' + str(config_dict['env']['size_iterable'][0])+'x'+str(config_dict['env']['size_iterable'][0])+' p='+str(config_dict['env']['p'])+ ' '+config_dict['model_params']['learner_iterable'][0]
    except:
        title = config_dict['problem_name'] + ' ' + config_dict['model_params']['learner_iterable'][0]
    plot_vs_n_iters(experiment, config_dict, df=df_gamma, title=title)
    
    
def vary_size_plotter(df, config_dict, experiment):
    df_size = df.set_index(['size'])
    title = config_dict['problem_name'] + ' p='+str(config_dict['env']['p'])+ ' ' + config_dict['model_params']['learner_iterable'][0]
    plot_vs_n_iters(experiment, config_dict, df=df_size, title=title)