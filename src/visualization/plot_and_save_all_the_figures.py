import os
import warnings

import numpy as np
import pandas as pd

from src.experiments.utils import development_datasets, univariate_equal_length
from src.visualization.utils import (
    plot_and_save_critical_difference_diagram,
    plot_and_save_multiple_comparison_matrix,
    plot_and_save_pairwise_scatter_plot,
)

warnings.filterwarnings("ignore")


def get_results_main():
    """Gather all the results in a single dictionary.

    Returns
    -------
    dict
        Dictionary with all the results.
    """

    # Define the metrics and their abbreviations
    metrics = {
        "ami": "Adjusted mutual information",
        "ari": "Adjusted Rand index",
        "clacc": "Clustering accuracy",
        "nmi": "Normalized mutual information"
    }

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


def section_5_2():
    """Comparisons to other time series clustering algorithms with the same setup"""
    # Get the main results
    results_main = get_results_main()

    # Train-test #
    algorithms_train_test = [
        "CLUES-WEASEL", "KASBA", "R-Clust", "MSM", "PAM-MSM", "Shape-DBA"
    ]

    # Critical difference diagrams
    for datasets in ("evaluation", "all"):
        for metric in ("ami", "ari", "clacc", "nmi"):
            plot_and_save_critical_difference_diagram(
                results=results_main,
                experiments="main",
                split="train-test",
                metric=metric,
                datasets=datasets,
                algorithms=algorithms_train_test,
            )

    # Pairwise scatter plots
    for datasets in ("evaluation", "all"):
        for metric in ("ami", "ari", "clacc", "nmi"):
            for algorithm in ("PAM-MSM", "KASBA", "Shape-DBA"):
                plot_and_save_pairwise_scatter_plot(
                    results=results_main,
                    experiments="main",
                    split="train-test",
                    metric=metric,
                    datasets=datasets,
                    algorithm_a="CLUES-WEASEL",
                    algorithm_b=algorithm
                )

    # Multiple comparison matrices
    for datasets in ("evaluation", "all"):
        for metric in ("ami", "ari", "clacc", "nmi"):
            plot_and_save_multiple_comparison_matrix(
                results=results_main,
                experiments="main",
                split="train-test",
                metric=metric,
                datasets=datasets,
                algorithms=algorithms_train_test,
                font_size=18
            )

    # Combined #
    algorithms_combined = [
        "CLUES-WEASEL", "DBA", "k-shape", "KASBA", "MSM", "Shape-DBA"
    ]

    # Critical difference diagrams
    for datasets in ("evaluation", "all"):
        for metric in ("ami", "ari", "clacc", "nmi"):
            plot_and_save_critical_difference_diagram(
                results=results_main,
                experiments="main",
                split="combined",
                metric=metric,
                datasets=datasets,
                algorithms=algorithms_combined,
            )


def section_5_3_1():
    """Comparisons to RandomNet."""

    results_main = get_results_main()

    results_clues_weasel_randomnet = {
        "combined": {
            "Adjusted Rand index": pd.concat([
                results_main["combined"]["Adjusted Rand index"]["CLUES-WEASEL"],
                pd.read_excel(
                    os.path.join("comparisons", "combined", "ari_results.xlsx"), index_col=0, skiprows=1
                ).squeeze().rename("RandomNet")
            ], axis=1).dropna()
        }
    }
    development_datasets_randomnet = [
        "Strawberry",
        "SwedishLeaf",
        "Symbols",
        "SyntheticControl",
        "ToeSegmentation1",
        "ToeSegmentation2",
        "Trace",
        "TwoLeadECG",
        "TwoPatterns",
        "UMD",
        "UWaveGestureLibraryAll",
        "UWaveGestureLibraryX",
        "UWaveGestureLibraryY",
        "UWaveGestureLibraryZ",
        "Wafer",
        "Wine",
        "WordSynonyms",
        "Worms",
        "WormsTwoClass",
        "Yoga",
    ]

    evaluation_datasets_clues_weasel_randomnet = np.setdiff1d(
        univariate_equal_length,
        np.union1d(development_datasets, development_datasets_randomnet)
    )

    for datasets, datasets_name in zip(
        [evaluation_datasets_clues_weasel_randomnet, univariate_equal_length],
        ["evaluation_datasets", "all_datasets"]
    ):
        plot_and_save_pairwise_scatter_plot(
            results=results_clues_weasel_randomnet,
            experiments="additional",
            split="combined",
            metric="ari",
            datasets=datasets,
            algorithm_a="CLUES-WEASEL",
            algorithm_b="RandomNet",
            datasets_name=datasets_name,
        )


def section_5_3_2():
    """Comparisons to trained deep neural networks."""

    # Get CLUES-WEASEL results
    results_main = get_results_main()

    # Get deep neural network results
    df = pd.read_csv(
        os.path.join("comparisons", "train-test", "deep_learning_results.csv")
    )

    def get_model_name(series):
        if isinstance(series['clustering_loss'], float):
            clustering_loss = "None"
        else:
            clustering_loss = series['clustering_loss']
        return f"{series['encoder_architecture']}_{series['encoder_loss']}_{clustering_loss}"

    df["model"] = df.apply(get_model_name, axis=1)

    df = df[df["dataset_name"].isin(univariate_equal_length)]

    df_acc = pd.pivot_table(df, values="acc", index="dataset_name", columns="model")

    best_deep_algorithms_acc = (
        df_acc.dropna(axis=1, thresh=92).mean(axis=0).nlargest(3).index.tolist()
        + df_acc.dropna(axis=1, thresh=92).rank(axis=1, ascending=False).mean(axis=0).nsmallest(3).index.tolist()
    )
    best_deep_algorithms_acc = list(set(best_deep_algorithms_acc))

    best_deep_algorithms_acc_no_na = (
        df_acc.dropna(axis=1, how="any").mean(axis=0).nlargest(3).index.tolist()
        + df_acc.dropna(axis=1, how="any").rank(axis=1, ascending=False).mean(axis=0).nsmallest(3).index.tolist()
    )
    best_deep_algorithms_acc_no_na = list(set(best_deep_algorithms_acc_no_na))

    df_nmi = pd.pivot_table(df, values="nmi", index="dataset_name", columns="model")

    best_deep_algorithms_nmi = (
        df_nmi.dropna(axis=1, thresh=92).mean(axis=0).nlargest(3).index.tolist()
        + df_nmi.dropna(axis=1, thresh=92).rank(axis=1, ascending=False).mean(axis=0).nsmallest(3).index.tolist()
    )
    best_deep_algorithms_nmi = list(set(best_deep_algorithms_nmi))

    best_deep_algorithms_nmi_no_na = (
        df_nmi.dropna(axis=1, how="any").mean(axis=0).nlargest(3).index.tolist()
        + df_nmi.dropna(axis=1, how="any").rank(axis=1, ascending=False).mean(axis=0).nsmallest(3).index.tolist()
    )
    best_deep_algorithms_nmi_no_na = list(set(best_deep_algorithms_nmi_no_na))

    results_clues_weasel_deep = {
        "train-test": {
            "Clustering accuracy": pd.concat([
                results_main["train-test"]["Clustering accuracy"]["CLUES-WEASEL"],
                df_acc[list(set(best_deep_algorithms_acc_no_na) | set(best_deep_algorithms_acc))]
            ], axis=1),
            "Normalized mutual information": pd.concat([
                results_main["train-test"]["Normalized mutual information"]["CLUES-WEASEL"],
                df_nmi[list(set(best_deep_algorithms_nmi_no_na) | set(best_deep_algorithms_nmi))]
            ], axis=1)
        }
    }

    # Critical difference diagrams
    for datasets in ("evaluation", "all"):
        for metric, best_deep_algorithms in zip(
            ["clacc", "clacc", "nmi", "nmi"],
            [
                best_deep_algorithms_acc,
                best_deep_algorithms_acc_no_na,
                best_deep_algorithms_nmi,
                best_deep_algorithms_nmi_no_na
            ]
        ):
            plot_and_save_critical_difference_diagram(
                results=results_clues_weasel_deep,
                experiments="additional",
                split="train-test",
                metric=metric,
                datasets=datasets,
                algorithms=["CLUES-WEASEL"] + best_deep_algorithms
            )


def section_5_4_1():
    """Comparisons to other transformers."""

    transformers = {
        "HydraTransformer": "Hydra",
        "MultiRocket": "MultiROCKET",
        "QUANTTransformer": "QUANT",
        "RandomDilatedShapeletTransform": "RDST",
        "TSFresh": "TSFresh",
        "WEASELTransformerV2": "WEASEL 2.0",
    }

    metrics = [
        "Adjusted mutual information",
        "Adjusted Rand index",
        "Clustering accuracy",
        "Normalized mutual information",
    ]

    results_ablation_other_transformers = {
        "train-test": {
            metric: pd.concat([
                pd.read_csv(
                    os.path.join("results", "train-test", tnf, "scores.csv"),
                    index_col=0
                )[metric].rename(tnf_name) for tnf, tnf_name in transformers.items()
            ], axis=1)
            for metric in metrics
        }
    }

    for datasets in ("evaluation", "all"):
        for metric in ("ami", "ari", "clacc", "nmi"):
            plot_and_save_critical_difference_diagram(
                results=results_ablation_other_transformers,
                experiments="ablation",
                split="train-test",
                metric=metric,
                datasets=datasets,
                algorithms=list(transformers.values()),
            )

            plot_and_save_multiple_comparison_matrix(
                results=results_ablation_other_transformers,
                experiments="ablation",
                split="train-test",
                metric=metric,
                datasets=datasets,
                algorithms=list(transformers.values()),
                font_size=18
            )

    results_main = get_results_main()

    results_all = {
        "train-test": {
            metric: pd.concat([
                results_main["train-test"][metric].drop("CLUES-WEASEL", axis=1),
                results_ablation_other_transformers["train-test"][metric],
            ], axis=1)
            for metric in metrics
        }
    }

    for datasets in ("evaluation", "all"):
        for metric in ("ami", "ari", "clacc", "nmi"):
            plot_and_save_critical_difference_diagram(
                results=results_all,
                experiments="ablation",
                split="train-test",
                metric=metric,
                datasets=datasets,
                algorithms=["PAM-MSM", "Shape-DBA", "KASBA"] + list(transformers.values()),
            )

    for datasets in ("evaluation", "all"):
        for metric in ("ami", "ari", "clacc", "nmi"):
            plot_and_save_multiple_comparison_matrix(
                results=results_all,
                experiments="ablation",
                split="train-test",
                metric=metric,
                datasets=datasets,
                algorithms=["PAM-MSM", "Shape-DBA", "KASBA"] + list(transformers.values()),
                font_size=18
            )


def section_5_4_2():
    metrics = [
        "Adjusted mutual information",
        "Adjusted Rand index",
        "Clustering accuracy",
        "Normalized mutual information",
    ]

    algorithms_max_feature_count = {
        "WEASELTransformerV2(10000)+PCA(0.4)": "(10k, 40%)",
        "WEASELTransformerV2(5000)+PCA(0.3)": "(5k, 30%)",
        "WEASELTransformerV2(1000)+PCA(0.3)": "(1k, 30%)",
        "WEASELTransformerV2(500)+PCA(0.3)": "(500, 30%)",
        "WEASELTransformerV2(200)+PCA(0.3)": "(200, 30%)",
    }

    results_ablation_max_feature_count = {
        "train-test": {
            metric: pd.concat([
                pd.read_csv(
                    os.path.join("ablation_experiments", "results", key, "scores.csv"),
                    index_col=0
                )[metric].rename(value)
                for key, value in algorithms_max_feature_count.items()
            ] + [
                pd.read_csv(
                    os.path.join("results", "train-test", "WEASELTransformerV2", "scores.csv"),
                    index_col=0
                )[metric].rename("(30k, 20%)")
            ], axis=1)
            for metric in metrics
        }
    }

    for datasets in ("evaluation", "all"):
        for metric in ("ami", "ari", "clacc", "nmi"):
            plot_and_save_critical_difference_diagram(
                results=results_ablation_max_feature_count,
                experiments="ablation",
                split="train-test",
                metric=metric,
                datasets=datasets,
                algorithms=list(algorithms_max_feature_count.values()) + ["(30k, 20%)"],
            )

    for datasets in ("evaluation", "all"):
        for metric in ("ami", "ari", "clacc", "nmi"):
            plot_and_save_multiple_comparison_matrix(
                results=results_ablation_max_feature_count,
                experiments="ablation",
                split="train-test",
                metric=metric,
                datasets=datasets,
                algorithms=list(algorithms_max_feature_count.values()) + ["(30k, 20%)"],
                font_size=18
            )


if __name__ == "__main__":
    section_5_2()
    section_5_3_1()
    section_5_3_2()
    section_5_4_1()
    section_5_4_2()
