from df_data import get_data_by_file_index
from df_plotting import plot_model_diagnostics
from df_plotting import discount_function_plotter
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

    f = plt.figure(figsize=(18, 12))
    ax1 = f.add_subplot(121)
    ax2 = f.add_subplot(222)
    ax3 = f.add_subplot(224)

    # plot all the fits on top of each other
    discount_function_plotter(ax1, models, data, save_dir, file_index, export=False)

    WAIC = model_comparison_WAIC(ax2, models, save_dir, file_index, MODEL_NAME_MAP, export=False)
    ax2.set_title('WAIC')
    print(WAIC)

    LOO = model_comparison_LOO(ax3, models, save_dir, file_index, MODEL_NAME_MAP, export=False)
    ax3.set_title('LOO')
    print(LOO)

    plt.savefig(f'{save_dir}/{file_index}_comparison.pdf', format='pdf', bbox_inches='tight')

    plt.cla()

    # metric_results = model_comparison_metrics(models, save_dir, file_index)
    # print(metric_results)

    plt.cla()
