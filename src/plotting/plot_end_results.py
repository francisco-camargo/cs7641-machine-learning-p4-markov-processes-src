import matplotlib.pyplot as plt

from src.plotting.plotting_helper import get_transparent_color

fontsize = 9
fontsize_ticks = fontsize - 2
fig_dim_x = 3.2
fig_dim_y = fig_dim_x * 0.75
alpha = 0.2

def generic_plotter(
            df,
            independent_variable:str,
            dependent_variable:str,
            dependent_variable_halfband:str,
            xlim=None,
            xscale=None,
            ylim=None,
            yscale=None,
            title:str=None,
            legend_loc:str='best',
            show:bool=False,
            ):
    
    unique_experiments = df.index.unique()
    fig = plt.figure()
    fig.set_size_inches(fig_dim_x, fig_dim_y)
    for idx, u_exp in enumerate(unique_experiments):
        df_temp = df.xs(u_exp)
        x       = df_temp[independent_variable]
        y       = df_temp[dependent_variable]
        band    = df_temp[dependent_variable_halfband]
        
        # Plot
        p     = plt.plot(x, y, label=str(u_exp))
        color = get_transparent_color(plot_object=p)
        plt.fill_between(x,y+band,y-band, color=color)
        
    if xscale: plt.xscale(xscale)
    if yscale: plt.yscale(yscale)
    plt.xlabel(independent_variable, fontsize=fontsize)
    plt.ylabel(dependent_variable, fontsize=fontsize)
    if xlim: plt.xlim(xlim)
    if ylim: plt.ylim(ylim)
    plt.tick_params(direction='in', which='both')
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.legend(loc=legend_loc, fontsize=fontsize_ticks, title=', '.join(df.index.names), title_fontsize=fontsize_ticks)
    plt.title(title, fontsize=fontsize)
    plt.tight_layout(pad=0)
    # fig.patch.set_facecolor('xkcd:mint green') # use to debug image sizing
    
    if show: plt.show()
    return plt
        