"""Run the ablation experiments."""

import os
import warnings

import numpy as np
import pandas as pd
from aeon.classification.dictionary_based._weasel_v2 import WEASELTransformerV2
from aeon.datasets import load_classification
from aeon.datasets.tsc_datasets import univariate_equal_length
from aeon.transformations.collection import Normalizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.experiments.utils import metric_names_abbreviations, scores, seeds

max_feature_counts = [5_000, 1_000, 500, 100]

# Hyperparameter optimization results
total_explained_variance_ratios = {
     "WEASELTransformerV2(10000)+PCA(0.4)": {
        "max_feature_count": 10000, "total_explained_variance_ratio": 0.4
    },
    "WEASELTransformerV2(5000)+PCA(0.3)": {
        "max_feature_count": 5000, "total_explained_variance_ratio": 0.3
    },
    "WEASELTransformerV2(1000)+PCA(0.3)": {
        "max_feature_count": 1000, "total_explained_variance_ratio": 0.3
    },
    "WEASELTransformerV2(500)+PCA(0.3)": {
        "max_feature_count": 500, "total_explained_variance_ratio": 0.3
    },
    "WEASELTransformerV2(200)+PCA(0.3)": {
        "max_feature_count": 200, "total_explained_variance_ratio": 0.3
    },
}

# Ignore warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    # Save the results in a dictionary
    results = {}

    for transformer_name in total_explained_variance_ratios:
        results[transformer_name] = {}

    # Get the number of development datasets
    n_datasets = len(univariate_equal_length)

    # For each data set
    for i, dataset_name in enumerate(sorted(univariate_equal_length)):

        # Print the current dataset
        print(f"[{i + 1:>{len(str(n_datasets))}}/{n_datasets}] {dataset_name}")

        for transformer_name in total_explained_variance_ratios:
            results[transformer_name][dataset_name] = {}

        # Load and normalize the dataset(s)
        X_train, _ = load_classification(dataset_name, split="train", return_metadata=False)
        X_test, labels_true = load_classification(dataset_name, split="test", return_metadata=False)
        X_train_normalized = Normalizer().transform(X_train)
        X_test_normalized = Normalizer().transform(X_test)

        # Encode the labels and get the true number of clusters
        labels_true = LabelEncoder().fit_transform(labels_true)
        n_clusters = np.unique(labels_true).size

        # For each RNG seed
        for seed in seeds:

            print(f"seed = {seed}")

            # For each maximum feature count
            for transformer_name in total_explained_variance_ratios:

                print(transformer_name)

                max_feature_count = total_explained_variance_ratios[transformer_name]["max_feature_count"]
                total_explained_variance_ratio = (
                    total_explained_variance_ratios[transformer_name]["total_explained_variance_ratio"]
                )

                tnf = WEASELTransformerV2(
                    max_feature_count=max_feature_count, feature_selection="random", random_state=seed, n_jobs=-1
                )

                pipeline = make_pipeline(tnf, VarianceThreshold(), StandardScaler(), PCA(random_state=seed))
                kmeans = KMeans(n_clusters, random_state=seed)

                # We have to add fake labels in WEASELTransformerV2.fit_transform() in aeon==1.3.0
                fake_labels = np.zeros(X_train_normalized.shape[0])
                X_train_pca = pipeline.fit_transform(X_train_normalized, fake_labels)

                X_test_pca = pipeline.transform(X_test_normalized)
                cumulative_explained_variance_ratio = np.cumsum(pipeline[-1].explained_variance_ratio_)
                if total_explained_variance_ratio == 1.0:
                    n_components = X_train_pca.shape[1]
                else:
                    n_components = max(
                        2, np.nonzero(cumulative_explained_variance_ratio >= total_explained_variance_ratio)[0][0]
                    )
                labels_pred = kmeans.fit(X_train_pca[:, :n_components]).predict(X_test_pca[:, :n_components])

                results[transformer_name][dataset_name][f"seed={seed}"] = scores(labels_true, labels_pred)

    # Create the split subdirectory if it does not exist
    if not os.path.isdir(os.path.join("ablation_experiments", "results", transformer_name)):
        os.makedirs(os.path.join("ablation_experiments", "results", transformer_name))

    metrics = list(metric_names_abbreviations.keys())

    # Save the results
    for transformer_name in total_explained_variance_ratios:
        # Create the split subdirectory if it does not exist
        if not os.path.isdir(os.path.join("ablation_experiments", "results", transformer_name)):
            os.makedirs(os.path.join("ablation_experiments", "results", transformer_name))

        df = pd.DataFrame(results[transformer_name]).T
        df = pd.concat([df.map(lambda x: x[metric]).mean(axis=1) for metric in metrics], axis=1)
        df.columns = metrics
        df.to_csv(os.path.join("ablation_experiments", "results", transformer_name, "scores.csv"))
