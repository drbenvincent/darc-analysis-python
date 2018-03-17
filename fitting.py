from df_data import get_data_by_file_index
from df_plotting import plot_model_diagnostics
from model_comparison import compare
#from IPython.display import display
import pandas as pd


def fit(model_classes, rawdata, expt_data, MODEL_NAME_MAP, save_dir='temp'):
    """Run parameter esimtation for models and data.
    - We will save all analyses in a folder called save_dir
    - We will save fits of all models in a participant subfolder
    """

    model_names = [model().__class__.__name__ for model in model_classes]
    print(model_names)
    META = pd.DataFrame()

    for file_index in expt_data.index:
        save_dir_subfolder = f'{save_dir}/{file_index}/'
        data = get_data_by_file_index(rawdata, file_index)

        # create a list of new model instances, used to fit this data
        # **** NOTE **** we are overriding each `models` for each data
        models = [model() for model in model_classes]

        for model in models:
            model.sample_posterior(data)
            model.results = expt_data # <----- TODO: check this
            plot_model_diagnostics(model, save_dir_subfolder, file_index)

        # Create a multi-level dataframe for point estimates for all Models.
        # Just one row, corresponding to the current file
        point_estimates_all_models = [model.point_estimates for model in models]
        # create a dataframe with MultiIndex columns
        results_this_file = pd.concat(point_estimates_all_models,
                                      axis=1,
                                      keys=model_names,
                                      names=['model', 'measure'])
        # append to large meta table
        META = pd.concat([META, results_this_file])

        # MODEL COMPARISON STUFF HERE -----------------------------------------
        compare(models, data, save_dir, file_index, MODEL_NAME_MAP)

    # ensure the index values are correct.
    META = META.reset_index()
    META.drop('index', axis=1, inplace=True)

    results = pd.concat([expt_data, META], axis=1)

    # export results
    full_save_path = f"{save_dir}/results.csv"
    print(f"Saving results to: {full_save_path}")
    results.to_csv(full_save_path)

    return results
