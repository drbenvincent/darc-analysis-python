import matplotlib.pyplot as plt
import pymc3 as pm
import seaborn as sns
import os as os
#from df_data import _data_df2dict, longest_delay


def plot_data(df, ax):
    ax.scatter(x=df.DB, y=df.A / df.B, marker='o', c=df.R, edgecolors='k')
    # constrain y-axis
    data_max_a_over_b = max(df.A / df.B,)
    ax.set_ylim([0, min(2, data_max_a_over_b+0.1)])


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def discount_function_plotter(models, data, path, file_id, export=True):
    """Plot data, and discount functions for as many models as we're given"""

    # colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9',
    #           'C10', 'C11', 'C12', 'C13', 'C14']
    fig, ax = plt.subplots(figsize=(14, 10))

    col_func = get_cmap(len(models))

    # 1: plot discount functions
    for i, model in enumerate(models):
        model.plot(data, ax, col=col_func(i))

    # 2: plot data
    plot_data(data, ax)

    # 3: format axes
    ax.legend()
    ax.set_xlabel('delay (days)')
    ax.set_ylabel('discount fraction')

    # 4: save
    if export is True:
        plt.savefig(f'{path}/{file_id}_df_comparison.pdf', format='pdf', bbox_inches='tight')


def plot_model_diagnostics(model, save_dir, file_id, export=True):
    """generate and export a range of diagnostic plots for a given model"""

    # ensure folder exists
    if export is True:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    model_name = model.__class__.__name__

    trace_df = pm.trace_to_dataframe(model.trace, varnames=model.df_params)

    sns.pairplot(trace_df)
    if export is True:
        plt.savefig(save_dir + f'{model_name}_{file_id}_pairplot.pdf',
                    format='pdf', bbox_inches='tight')
        plt.cla()

    pm.traceplot(model.trace, varnames=model.df_params)
    if export is True:
        plt.savefig(save_dir + f'{model_name}_{file_id}_traceplot.pdf',
                    format='pdf', bbox_inches='tight')
        plt.cla()

    pm.autocorrplot(model.trace, varnames=model.df_params)
    if export is True:
        plt.savefig(save_dir + f'{model_name}_{file_id}_autocorrplot.pdf',
                    format='pdf', bbox_inches='tight')
        plt.cla()

    pm.forestplot(model.trace, varnames=model.df_params)
    if export is True:
        plt.savefig(save_dir + f'{model_name}_{file_id}_forestplot.pdf',
                    format='pdf', bbox_inches='tight')
        plt.cla()

    # close all figs, otherwise we can run out of memory
    plt.close("all")
