import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import seaborn as sns
import os as os


def plot_data(df, ax):
    ax.scatter(x=df.DB, y=df.A / df.B, marker='o', c=df.R, edgecolors='k')
    # constrain y-axis
    data_max_a_over_b = max(df.A / df.B,)
    ax.set_ylim([0, min(2, data_max_a_over_b+0.1)])


def plot_discount_function_lines(ax, delay, df_matrix):
    for n in range(0, df_matrix.shape[0]):
        plt.plot(delay, df_matrix[n, :], color='k', alpha=0.1)


def plot_discount_functions_region(ax, delays, df_matrix, alpha=0.05, col='r', label=None, plotCI=True):
    curve_mean = df_matrix.mean(axis=1)
    # curve_median = np.median(df_matrix, axis=1)

    if plotCI:
        percentiles = 100 * np.array([alpha / 2., 1. - alpha / 2.])
        hpd = np.percentile(df_matrix, percentiles, axis=1)
        ax.fill_between(delays, hpd[0], hpd[1], facecolor=col, alpha=0.25)

    plt.plot(delays, curve_mean, color=col, label=label, lw=2.)


def df_comparison(models, data, path, file_id, plotCI=True, export=True):
    """Plot data and the posterior predictions for mulitple models"""

    # plot bar chart of metrics ------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 10))
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']

    for i, model in enumerate(models):
        delays, df_pp_matrix = model.df_posterior_prediction()
        plot_discount_functions_region(ax, delays, df_pp_matrix, col=colors[i], label = model.__class__.__name__, plotCI=plotCI)

    ax.legend()
    plot_data(data, ax)
    ax.set_xlabel('delay (days)')
    ax.set_ylabel('discount fraction')

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
