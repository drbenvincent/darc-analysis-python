import pymc3 as pm
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os as os


def model_comparison_WAIC(ax, models, path, file_id, MODEL_NAME_MAP, should_plot=True, export=True):
    """Conduct some model comparison using WAIC, give a list of models"""
    traces = [model.trace for model in models]
    models = [model.model for model in models]
    WAIC = (pm.compare(traces, models).rename(index=MODEL_NAME_MAP))
    if should_plot is True:
        pm.compareplot(WAIC, ax=ax)
        if export is True:
            plt.savefig(f'{path}/{file_id}_WAIC.pdf',
                        format='pdf', bbox_inches='tight')
        #plt.cla()
    return WAIC


def model_comparison_LOO(ax, models, path, file_id, MODEL_NAME_MAP, should_plot=True, export=True):
    """Conduct some model comparison using LOO, give a list of models"""
    traces = [model.trace for model in models]
    models = [model.model for model in models]
    LOO = (pm.compare(traces, models, ic='LOO').rename(index=MODEL_NAME_MAP))
    if should_plot is True:
        pm.compareplot(LOO, ax=ax)
        if export is True:
            plt.savefig(f'{path}/{file_id}_LOO.pdf',
                        format='pdf', bbox_inches='tight')
        #plt.cla()
    return LOO


def model_comparison_metrics(models, save_dir, file_id, should_plot=True, export=True):
    """For a set of models, construct a table of metrics and plot it"""

    if export is True:
        directory = f'{save_dir}/'
        if not os.path.exists(directory):
            os.makedirs(directory)

    log_loss_median = [np.median(model.metrics['log_loss']) for model in models]
    model_names = [model.__class__.__name__ for model in models]

    output = pd.DataFrame(
        {'name': model_names,
         'log_loss_median': log_loss_median
         }).set_index('name')

    if should_plot is True:
        f, ax = plt.subplots(1, 1, figsize=(8, 6))

        log_loss_plot = sns.barplot(model_names, log_loss_median, palette="Set3", ax=ax)
        ax.set_ylabel("log loss")

        log_loss_plot.set_xticklabels(log_loss_plot.get_xticklabels(), rotation=-90)

        if export is True:
            plt.savefig(f'{directory}/{file_id}_model_metric.pdf', format='pdf', bbox_inches='tight')

        f.clear()

    # TODO: return as a pandas dataframe
    return output
