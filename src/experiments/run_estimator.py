"""Run EKOTAM on all the UCR univariate equal-length datasets."""

import os

import numpy as np
import pandas as pd
from aeon.classification.dictionary_based._weasel_v2 import WEASELTransformerV2
from aeon.datasets import load_classification
from aeon.datasets.tsc_datasets import univariate_equal_length
from aeon.transformations.collection import Normalizer
from aeon.transformations.collection.convolution_based import MultiRocket
from aeon.transformations.collection.convolution_based._hydra import HydraTransformer
from aeon.transformations.collection.feature_based import TSFresh
from aeon.transformations.collection.interval_based._quant import QUANTTransformer
from aeon.transformations.collection.shapelet_based import RandomDilatedShapeletTransform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.experiments.utils import metric_names_abbreviations, scores, seeds


def run_all_datasets(split, transformer_name, total_explained_variance_ratio):
    """Run an algorithm on all the 112 univariate equal length datasets.

    Parameters
    ----------
    split = {"train-test", "combined"}
        Splitting method for the dataset.

    transformer_name : str
        Name of the time series transformer.

    total_explained_variance_ratio : float
        Total explained variance ratio. Used to select the number of components.
    """

    # Save the results in a dictionary
    results = {}

    # Get the number of development datasets
    n_datasets = len(univariate_equal_length)

    # For each data set
    for i, dataset_name in enumerate(sorted(univariate_equal_length)):

        # Print the current dataset
        print(f"{transformer_name} [{i + 1:>{len(str(n_datasets))}}/{n_datasets}] {dataset_name}")

        results[dataset_name] = {}

        # Load and normalize the dataset(s)
        if split == "train-test":
            X_train, _ = load_classification(dataset_name, split="train", return_metadata=False)
            X_test, labels_true = load_classification(dataset_name, split="test", return_metadata=False)
            X_train_normalized = Normalizer().transform(X_train)
            X_test_normalized = Normalizer().transform(X_test)
        else:
            X, labels_true = load_classification(dataset_name, split=None, return_metadata=False)
            X_normalized = Normalizer().transform(X)

        # Encode the labels and get the true number of clusters
        labels_true = LabelEncoder().fit_transform(labels_true)
        n_clusters = np.unique(labels_true).size

        # For each RNG seed
        for seed in seeds:

            print(f"seed = {seed}")

            transformers = {
                "WEASELTransformerV2": WEASELTransformerV2(feature_selection="random", random_state=seed, n_jobs=-1),
                "MultiRocket": MultiRocket(random_state=seed, n_jobs=-1),
                "HydraTransformer": HydraTransformer(output_type="numpy", random_state=seed, n_jobs=-1),
                "RandomDilatedShapeletTransform": RandomDilatedShapeletTransform(random_state=seed, n_jobs=-1),
                "QUANTTransformer": QUANTTransformer(),
                "TSFresh": TSFresh(n_jobs=-1),
            }

            tnf = transformers[transformer_name]

            if split == "train-test":
                pipeline = make_pipeline(tnf, VarianceThreshold(), StandardScaler(), PCA(random_state=seed))
                if transformer_name == "WEASELTransformerV2":
                    # We have to add fake labels in WEASELTransformerV2.fit_transform() in aeon==1.3.0
                    fake_labels = np.zeros(X_train_normalized.shape[0])
                    X_train_pca = pipeline.fit_transform(X_train_normalized, fake_labels)
                else:
                    X_train_pca = pipeline.fit_transform(X_train_normalized)
                X_test_pca = pipeline.transform(X_test_normalized)
                cumulative_explained_variance_ratio = np.cumsum(pipeline[-1].explained_variance_ratio_)
                if total_explained_variance_ratio == 1.0:
                    n_components = X_train_pca.shape[1]
                else:
                    n_components = max(
                        2, np.nonzero(cumulative_explained_variance_ratio >= total_explained_variance_ratio)[0][0]
                    )
                kmeans = KMeans(n_clusters, random_state=seed)
                labels_pred = kmeans.fit(X_train_pca[:, :n_components]).predict(X_test_pca[:, :n_components])
            else:
                pipeline = make_pipeline(tnf, VarianceThreshold(), StandardScaler())
                if transformer_name == "WEASELTransformerV2":
                    fake_labels = np.zeros(X_normalized.shape[0])
                    X_tnf = pipeline.fit_transform(X_normalized, fake_labels)
                else:
                    X_tnf = pipeline.fit_transform(X_normalized)
                try:
                    pca = PCA(random_state=seed)
                    X_pca = pca.fit_transform(X_tnf)
                except:  # noqa: E722
                    pca = IncrementalPCA(batch_size=100)
                    X_pca = pca.fit_transform(X_tnf)
                cumulative_explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
                n_components = max(
                    2, np.nonzero(cumulative_explained_variance_ratio >= total_explained_variance_ratio)[0][0]
                )
                kmeans = KMeans(n_clusters, random_state=seed)
                labels_pred = kmeans.fit_predict(X_pca[:, :n_components])
            results[dataset_name][f"seed={seed}"] = scores(labels_true, labels_pred)

    # Create the split subdirectory if it does not exist
    if not os.path.isdir(os.path.join("results", split, transformer_name)):
        os.makedirs(os.path.join("results", split, transformer_name))

    metrics = list(metric_names_abbreviations.keys())
    df = pd.DataFrame(results).T
    df = pd.concat([df.map(lambda x: x[metric]).mean(axis=1) for metric in metrics], axis=1)
    df.columns = metrics
    df.to_csv(os.path.join("results", split, transformer_name, "scores.csv"))
