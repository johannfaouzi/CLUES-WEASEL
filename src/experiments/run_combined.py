"""Run EKOTAM on all the UCR univariate equal-length datasets using train-test splits."""

import warnings

from src.experiments.run_estimator import run_all_datasets
from src.experiments.utils import total_explained_variance_ratios

# Ingore warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    run_all_datasets("combined", "WEASELTransformerV2", total_explained_variance_ratios["WEASELTransformerV2"])
