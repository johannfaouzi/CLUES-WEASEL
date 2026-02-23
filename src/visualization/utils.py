import os

import pandas as pd
from aeon.visualisation import plot_critical_difference

from src.experiments.utils import development_datasets, evaluation_datasets, univariate_equal_length
from src.visualization.multi_comparison_matrix import compare
from src.visualization.scatter_plot import plot_pairwise_scatter

metrics = {
    "ami": "Adjusted mutual information",
    "ari": "Adjusted Rand index",
    "clacc": "Clustering accuracy",
    "nmi": "Normalized mutual information"
}


def get_results_main():
    """Gather all the results in a single dictionary.

    Returns
    -------
    dict
        Dictionary with all the results.
    """

    # Initialize a dictionary to save all the results
    results = {}

    # Train-test #
    results["train-test"] = {}

    # Get the results for CLUES-WEASEL
    df_train_test_weasel = pd.read_csv(
        os.path.join("results", "train-test", "WEASELTransformerV2", "scores.csv"),
        index_col=0
    )

    # Get the results for the other algorithms
    temp_train_test = {}
    for metric_abb, metric in metrics.items():
        temp_train_test[metric] = pd.read_csv(
            os.path.join("comparisons", "train-test", f"kasba_{metric_abb}_mean.csv"),
            index_col=0
        )

    # Concat the results
    for metric in metrics.values():
        df_temp = pd.concat(
            [df_train_test_weasel[metric], temp_train_test[metric]], axis=1
        )
        df_temp.columns = ["CLUES-WEASEL"] + temp_train_test[metric].columns.to_list()
        results["train-test"][metric] = df_temp
        del df_temp

    # Delete the temporary dictionary
    del temp_train_test

    # Combined #
    results["combined"] = {}

    # Get the results for CLUES-WEASEL
    df_combined_weasel = pd.read_csv(
        os.path.join("results", "combined", "WEASELTransformerV2", "scores.csv"),
        index_col=0
    )

    # Get the results for the other algorithms
    temp_combined = {}
    for metric_abb, metric in metrics.items():
        temp_combined[metric] = pd.read_csv(
            os.path.join("comparisons", "combined", f"kasba_{metric_abb}_mean.csv"),
            index_col=0
        )

    # Concat the results
    for metric in metrics.values():
        df_temp = pd.concat(
            [df_combined_weasel[metric], temp_combined[metric]], axis=1
        )
        df_temp.columns = ["CLUES-WEASEL"] + temp_combined[metric].columns.to_list()
        results["combined"][metric] = df_temp
        del df_temp

    # Delete the temporary dictionary
    del temp_combined

    return results


def plot_and_save_critical_difference_diagram(
    results, experiments, split, metric, datasets, algorithms, datasets_name=None, **kwargs
):
    """Plot and save the critical difference diagram.

    Parameters
    ----------
    results : dict
        Dictionary with all the results

    experiments : {"main", "additional", "ablation"}
        Type of experiments.

    split = {"train-test", "combined"}
        Splitting method.

    metric : {"ami", "ari", "clacc", "nmi"}
        Metric used to compare the algorithms:
            - "ami" is for adjusted mutual information
            - "ari" is for adjusted Rand index
            - "clacc" is for clustering accuracy
            - "nmi" is for normalized mutual information

    datasets : {"all", "development", "evaluation"}
        Datasets used to generate the figure.

    algorithms : list[str]
        Algorithms to compare.

    datasets_name : str or None, default = None
        Name of the list of datasets. Only used if not None.

    kwargs
        Keyword arguments passed to plot_critical_difference().
    """
    if not (isinstance(experiments, str) and experiments in ("main", "additional", "ablation")):
        raise ValueError("Invalid value for 'experiments'.")

    if not (isinstance(split, str) and split in ("train-test", "combined")):
        raise ValueError("Invalid value for 'split'.")

    if not (isinstance(metric, str) and metric in metrics):
        raise ValueError("Invalid value for 'metric'.")

    df = results[split][metrics[metric]]

    if isinstance(datasets, str):
        if datasets == "all":
            datasets_ = univariate_equal_length
        elif datasets == "development":
            datasets_ = development_datasets
        elif datasets == "evaluation":
            datasets_ = evaluation_datasets
        else:
            raise ValueError("Invalid string for 'datasets'.")
    else:
        datasets_ = datasets

    if not (isinstance(algorithms, (list, tuple, set)) and all(isinstance(name, str) for name in algorithms)):
        raise TypeError("'algorithms' must be an iterable of strings.")
    if not all(algorithm in df.columns for algorithm in algorithms):
        raise ValueError("At least one algorithm name is not valid.")

    # Create the directory in which the figure will be saved
    directory = os.path.join(
        "figures",
        "critical_difference_diagrams",
        f"{experiments}",
        f"{split}",
        f"{'_vs_'.join(algorithms)}".replace("%", "p"),
        f"{datasets}_datasets" if datasets_name is None else str(datasets_name),
        f"{'_'.join(word.lower() for word in metrics[metric].split(" "))}"
    )
    if not os.path.isdir(directory):
        os.makedirs(directory)

    # Create the critical difference diagram and save it
    df = df.loc[datasets_, algorithms].dropna(axis=0)
    fig, _ = plot_critical_difference(df.to_numpy(), df.columns, **kwargs)
    fig.savefig(os.path.join(directory, f"figure(n_datasets={df.shape[0]}).pdf"), bbox_inches="tight")


def plot_and_save_pairwise_scatter_plot(
    results, experiments, split, metric, datasets,
    algorithm_a, algorithm_b, name_algorithm_a=None, name_algorithm_b=None,
    datasets_name=None
):
    """Plot and save the pairwise scatter plot.

    Parameters
    ----------
    results : dict
        Dictionary with all the results

    experiments : {"main", "additional", "ablation"}
        Type of experiments.

    split = {"train-test", "combined"}
        Splitting method.

    metric : {"ami", "ari", "clacc", "nmi"}
        Metric used to compare the algorithms:
            - "ami" is for adjusted mutual information
            - "ari" is for adjusted Rand index
            - "clacc" is for clustering accuracy
            - "nmi" is for normalized mutual information

    datasets : {"all", "development", "evaluation"}
        Datasets used to generate the figure.

    algorithm_a : str
        First algorithm. It should be the name of a column in the pandas.DataFrame.

    algorithm_b : str
        Second algorithm. It should be the name of a column in the pandas.DataFrame.

    name_algorithm_a = str, default = None
        Name of the first algorithm in the figure. If None, `algorithm_a` is used.

    name_algorithm_b = str, default = None
        Name of the second algorithm in the figure. If None, `algorithm_b` is used.

    datasets_name : str or None, default = None
        Name of the list of datasets. Only used if not None.
    """

    if not (isinstance(experiments, str) and experiments in ("main", "additional", "ablation")):
        raise ValueError("Invalid value for 'experiments'.")

    if not (isinstance(split, str) and split in ("train-test", "combined")):
        raise ValueError("Invalid value for 'split'.")

    if not (isinstance(metric, str) and metric in metrics):
        raise ValueError("Invalid value for 'metric'.")

    if isinstance(datasets, str):
        if datasets == "all":
            datasets_ = univariate_equal_length
        elif datasets == "development":
            datasets_ = development_datasets
        elif datasets == "evaluation":
            datasets_ = evaluation_datasets
        else:
            raise ValueError("Invalid string for 'datasets'.")
    else:
        datasets_ = datasets

    # Create the directory in which the figure will be saved
    directory = os.path.join(
        "figures",
        "scatter_plots",
        f"{experiments}",
        f"{split}",
        f"{algorithm_a}_vs_{algorithm_b}".replace("%", "p"),
        f"{datasets}_datasets" if datasets_name is None else str(datasets_name),
        f"{'_'.join(word.lower() for word in metrics[metric].split(" "))}"
    )
    if not os.path.isdir(directory):
        os.makedirs(directory)

    metric_full_name = metrics[metric]

    # Create the scatter plot and save it
    df = results[split][metrics[metric]].loc[datasets_, [algorithm_a, algorithm_b]].dropna(axis=0)
    fig, _ = plot_pairwise_scatter(
        results_a=df[algorithm_a].to_numpy(),
        results_b=df[algorithm_b].to_numpy(),
        method_a=str(name_algorithm_a) if name_algorithm_a is not None else algorithm_a,
        method_b=str(name_algorithm_b) if name_algorithm_b is not None else algorithm_b,
        metric=metric_full_name[0].lower() + metric_full_name[1:]
    )
    fig.savefig(os.path.join(directory, f"figure(n_datasets={df.shape[0]}).pdf"), bbox_inches="tight")


def plot_and_save_multiple_comparison_matrix(
    results, experiments, split, metric, datasets, algorithms, datasets_name=None, **kwargs
):
    """Plot and save the multiple comparison matrix.

    Parameters
    ----------

    results : dict
        Dictionary with all the results

    experiments : {"main", "additional", "ablation"}
        Type of experiments.

    split = {"train-test", "combined"}
        Splitting method.

    metric : {"ami", "ari", "clacc", "nmi"}
        Metric used to compare the algorithms:
            - "ami" is for adjusted mutual information
            - "ari" is for adjusted Rand index
            - "clacc" is for clustering accuracy
            - "nmi" is for normalized mutual information

    datasets : {"all", "development", "evaluation"}
        Datasets used to generate the figure.

    algorithms : list[str]
        Algorithms to compare.

    datasets_name : str or None, default = None
        Name of the list of datasets. Only used if not None.

    kwargs
        Keyword arguments passed to compare().
    """
    if not (isinstance(experiments, str) and experiments in ("main", "additional", "ablation")):
        raise ValueError("Invalid value for 'experiments'.")

    if not (isinstance(split, str) and split in ("train-test", "combined")):
        raise ValueError("Invalid value for 'split'.")

    if not (isinstance(metric, str) and metric in metrics):
        raise ValueError("Invalid value for 'metric'.")

    df = results[split][metrics[metric]]

    if isinstance(datasets, str):
        if datasets == "all":
            datasets_ = univariate_equal_length
        elif datasets == "development":
            datasets_ = development_datasets
        elif datasets == "evaluation":
            datasets_ = evaluation_datasets
        else:
            raise ValueError("Invalid string for 'datasets'.")
    else:
        datasets_ = datasets

    if not (isinstance(algorithms, (list, tuple, set)) and all(isinstance(name, str) for name in algorithms)):
        raise TypeError("'algorithms' must be an iterable of strings.")
    if not all(algorithm in df.columns for algorithm in algorithms):
        raise ValueError("At least one algorithm name is not valid.")

    # Create the directory in which the figure will be saved
    directory = os.path.join(
        "figures",
        "multiple_comparison_matrices",
        f"{experiments}",
        f"{split}",
        f"{'_vs_'.join(algorithms)}".replace("%", "p"),
        f"{datasets}_datasets" if datasets_name is None else str(datasets_name),
        f"{'_'.join(word.lower() for word in metrics[metric].split(" "))}"
    )
    if not os.path.isdir(directory):
        os.makedirs(directory)

    df = df.loc[datasets_, algorithms].dropna(axis=0)
    compare(df, pdf_savename=os.path.join(directory, f"figure(n_datasets={df.shape[0]})"), **kwargs)
