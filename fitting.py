from df_data import get_data_by_file_index
from df_plotting import plot_model_diagnostics
from df_plotting import df_comparison
from model_comparison import model_comparison_metrics, model_comparison_LOO, model_comparison_WAIC

import matplotlib.pyplot as plt


def fit(model_classes, rawdata, expt_data, MODEL_NAME_MAP, save_dir='temp'):
    """Run parameter esimtation for models and data.
    - We will save all analyses in a folder called save_dir
    - We will save fits of all models in a participant subfolder
    """

    for file_index in expt_data.index:
        save_dir_subfolder = f'{save_dir}/{file_index}/'
        data = get_data_by_file_index(rawdata, file_index)

        # create a list of new model instances, used to fit this data
        # **** NOTE **** we are overriding each `models` for each data
        models = [model() for model in model_classes]

        for model in models:
            model.sample_posterior(data)
            model.results = expt_data
            # ** UPDATE model.results HERE
            plot_model_diagnostics(model, save_dir_subfolder, file_index)
            # TODO: save model fit here

        # MODEL COMPARISON STUFF HERE --------------------------------------
        compare(models, data, save_dir, file_index, MODEL_NAME_MAP)

    return models


def compare(models, data, save_dir, file_index, MODEL_NAME_MAP):
    """ For a list of models 'fitted' to one data file, conduct model
    comparison and save some plots. """

    # plot all the fits on top of each other
    df_comparison(models, data, save_dir, file_index)

    metric_results = model_comparison_metrics(models, save_dir, file_index)
    print(metric_results)

    WAIC = model_comparison_WAIC(models, save_dir, file_index, MODEL_NAME_MAP)
    print(WAIC)
    plt.cla()

    LOO = model_comparison_LOO(models, save_dir, file_index, MODEL_NAME_MAP)
    print(LOO)
    plt.cla()

    # close all figs, otherwise we can run out of memory
    #plt.close("all")
