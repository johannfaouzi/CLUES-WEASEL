"""Run EKOTAM on all the UCR univariate equal-length datasets using train-test splits."""

import warnings

from src.experiments.run_estimator import run_all_datasets
from src.experiments.utils import total_explained_variance_ratios

# Ignore warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    for transformer_name, total_explained_variance_ratio in total_explained_variance_ratios.items():
        run_all_datasets("train-test", transformer_name, total_explained_variance_ratio)
