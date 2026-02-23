import os

import matplotlib.pyplot as plt
import numpy as np
from aeon.classification.dictionary_based._weasel_v2 import WEASELTransformerV2
from aeon.datasets import load_classification
from aeon.transformations.collection import Normalizer
from aeon.transformations.collection.convolution_based import MultiRocket
from aeon.transformations.collection.convolution_based._hydra import HydraTransformer
from aeon.transformations.collection.feature_based import TSFresh
from aeon.transformations.collection.interval_based._quant import QUANTTransformer
from aeon.transformations.collection.shapelet_based import RandomDilatedShapeletTransform
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from umap import UMAP


def scatter_plot_extracted_features(dataset_name, random_state, s):

    transformers = {
        "WEASEL 2.0": WEASELTransformerV2(
            feature_selection="random", random_state=random_state, n_jobs=-1
        ),
        "MultiRocket": MultiRocket(random_state=random_state, n_jobs=-1),
        "Hydra": HydraTransformer(
            output_type="numpy", random_state=random_state, n_jobs=-1
        ),
        "RDST": RandomDilatedShapeletTransform(
            random_state=random_state, n_jobs=-1
        ),
        "QUANT": QUANTTransformer(),
        "TSFresh": TSFresh(n_jobs=-1),
    }

    total_explained_variance_ratios = {
        "WEASEL 2.0": 0.2,
        "MultiRocket": 0.9,
        "Hydra": 0.7,
        "RDST": 0.7,
        "QUANT": 0.7,
        "TSFresh": 1.0,
    }

    X_train, _ = load_classification(dataset_name, split="train", return_metadata=False)
    X_test, labels_true = load_classification(dataset_name, split="test", return_metadata=False)
    le = LabelEncoder()
    labels_true = le.fit_transform(labels_true)
    X_train_normalized = Normalizer().transform(X_train)
    X_test_normalized = Normalizer().transform(X_test)

    res = {}

    for transformer_name, transformer in transformers.items():
        total_explained_variance_ratio = total_explained_variance_ratios[transformer_name]
        pipeline = make_pipeline(transformer, VarianceThreshold(), StandardScaler(), PCA(random_state=42))
        if transformer_name == "WEASEL 2.0":
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

        if n_components > 2:
            umap = UMAP(n_jobs=1, random_state=random_state)
            X_test_pca_2d = umap.fit_transform(X_test_pca[:, :n_components])
        else:
            X_test_pca_2d = X_test_pca[:, :n_components]

        res[transformer_name] = X_test_pca_2d

    directory = os.path.join(
        "figures",
        "scatter_plots",
        "exploratory",
        "train-test",
    )
    if not os.path.isdir(directory):
        os.makedirs(directory)

    # Plot the figures
    plt.figure(figsize=(8, 10))
    for i, transformer_name in enumerate(transformers):
        plt.subplot(3, 2, i + 1)
        plt.scatter(
            x=res[transformer_name][:, 0],
            y=res[transformer_name][:, 1],
            s=s,
            c=np.array([f"C{k}" for k in range(le.classes_.size)])[labels_true]
        )
        plt.title(transformer_name)

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.savefig(
        os.path.join(directory, f"figure({dataset_name=!s}_{random_state=}).pdf"),
        bbox_inches="tight"
    )


if __name__ == "__main__":
    random_state = 42
    scatter_plot_extracted_features(dataset_name="Coffee", random_state=random_state, s=15)
    scatter_plot_extracted_features(dataset_name="Fish", random_state=random_state, s=3)
    scatter_plot_extracted_features(dataset_name="Car", random_state=random_state, s=3)
    scatter_plot_extracted_features(dataset_name="Symbols", random_state=random_state, s=1)
    scatter_plot_extracted_features(dataset_name="MiddlePhalanxTW", random_state=random_state, s=1)
