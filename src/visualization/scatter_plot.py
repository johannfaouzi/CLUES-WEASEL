# Adapted from https://github.com/aeon-toolkit/aeon/blob/main/aeon/visualisation/results/_scatter.py

import warnings

import numpy as np
from aeon.utils.validation._dependencies import _check_soft_dependencies


def plot_pairwise_scatter(
    results_a,
    results_b,
    method_a,
    method_b,
    metric="accuracy",
    lower_better=False,
    statistic_tests=True,
    title=None,
    figsize=(8, 8),
    color_palette="tab10",
    best_on_top=True,
):
    """Plot a scatter that compares datasets' results achieved by two methods.

    Parameters
    ----------
    results_a : np.array
        Scores (either accuracies or errors) per dataset for the first approach.
    results_b : np.array
        Scores (either accuracies or errors) per dataset for the second approach.
    method_a : str
        Method name of the first approach.
    method_b : str
        Method name of the second approach.
    metric : str, default = "accuracy"
        Metric to be used for the comparison.
    lower_better : bool, default = False
        If True, lower values are considered better, i.e. errors.
    statistic_tests : bool, default = True
        If True, paired ttest and wilcoxon p-values are shown in the bottom of the plot.
    title : str, default = None
        Title to be shown in the top of the plot.
    figsize : tuple, default = (10, 6)
        Size of the figure.
    color_palette : str, default = "tab10"
        Color palette to be used for the plot.
    best_on_top : bool, default=True
        If True, the estimator with better performance is placed on the y-axis (top).
        If False, the ordering is reversed.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Examples
    --------
    >>> from aeon.visualisation import plot_pairwise_scatter
    >>> from aeon.benchmarking.results_loaders import get_estimator_results_as_array
    >>> methods = ["InceptionTimeClassifier", "WEASEL-Dilation"]
    >>> results = get_estimator_results_as_array(estimators=methods)  # doctest: +SKIP
    >>> plot = plot_pairwise_scatter(  # doctest: +SKIP
    ...     results[0], methods[0], methods[1])  # doctest: +SKIP
    >>> plot.show()  # doctest: +SKIP
    >>> plot.savefig("scatterplot.pdf")  # doctest: +SKIP
    """
    _check_soft_dependencies("matplotlib", "seaborn")

    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.offsetbox import AnchoredText

    palette = sns.color_palette(color_palette, n_colors=3)

    if isinstance(results_a, list):
        results_a = np.array(results_a)
    if isinstance(results_b, list):
        results_b = np.array(results_b)

    if len(results_a.shape) != 1:
        raise ValueError("results_a must be a 1D array.")
    if len(results_b.shape) != 1:
        raise ValueError("results_b must be a 1D array.")

    if statistic_tests:
        fig, ax = plt.subplots(figsize=figsize, gridspec_kw=dict(bottom=0.2))
    else:
        fig, ax = plt.subplots(figsize=figsize)

    results_all = np.concatenate((results_a, results_b))
    min_value = (
        results_all.min() * 1.05
        if (results_all.min() < 0)
        else results_all.min() * 0.95
    )
    max_value = (
        results_all.max() * 1.05
        if (results_all.max() > 0)
        else results_all.max() * 0.95
    )

    x, y = [min_value, max_value], [min_value, max_value]
    ax.plot(x, y, color="black", alpha=0.5, zorder=1)

    # better estimator on top (y-axis)
    if (results_a.mean() <= results_b.mean() and not lower_better) or (
        results_a.mean() >= results_b.mean() and lower_better
    ):
        first = results_b
        first_method = method_b
        second = results_a
        second_method = method_a
    else:
        first = results_a
        first_method = method_a
        second = results_b
        second_method = method_b

    # if best_on_top is False, swap the ordering
    if not best_on_top:
        first, second = second, first
        first_method, second_method = second_method, first_method

    differences = [
        0 if i - j == 0 else (1 if i - j > 0 else -1) for i, j in zip(first, second)
    ]
    # This line helps displaying ties on top of losses and wins, as in general there
    # are less number of ties than wins/losses.
    differences, first, second = map(
        np.array,
        zip(*sorted(zip(differences, first, second), key=lambda x: -abs(x[0]))),
    )

    first_median = np.median(first)
    second_median = np.median(second)

    plot = sns.scatterplot(
        x=second,
        y=first,
        hue=differences,
        hue_order=[1, 0, -1] if lower_better else [-1, 0, 1],
        palette=palette,
        zorder=2,
    )

    # Draw the median value per method as a dashed line from 0 to the median value.
    ax.plot(
        [first_median, min_value] if not lower_better else [first_median, max_value],
        [first_median, first_median],
        linestyle="--",
        color=palette[2],
        zorder=3,
    )

    ax.plot(
        [second_median, second_median],
        [second_median, min_value] if not lower_better else [second_median, max_value],
        linestyle="--",
        color=palette[0],
        zorder=3,
    )

    legend_median = AnchoredText(
        "* Dashed lines represent the median",
        loc="lower right" if lower_better else "upper right",
        prop=dict(size=8),
        bbox_to_anchor=(1.01, 1.09 if lower_better else -0.09),
        bbox_transform=ax.transAxes,
    )
    ax.add_artist(legend_median)

    # Compute the W, T, and L per methods
    if lower_better:
        differences = [-i for i in differences]
        ax = plt.gca()
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.spines["top"].set_visible(True)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.spines["right"].set_visible(True)

    # Setting labels for x and y axis
    plot.set_ylabel(f"{first_method} {metric}\n(mean: {first.mean():.4f})", fontsize=13)
    plot.set_xlabel(
        f"{second_method} {metric}\n(mean: {second.mean():.4f})", fontsize=13
    )

    wins_A = losses_B = sum(i == 1 for i in differences)
    ties_A = ties_B = sum(i == 0 for i in differences)
    losses_A = wins_B = sum(i == -1 for i in differences)

    # Setting x and y limits
    plot.set_ylim(min_value, max_value)
    plot.set_xlim(min_value, max_value)

    # Remove legend
    plot.get_legend().remove()

    # Setting text with W, T and L for each method
    anc_text = AnchoredText(
        f"{first_method} wins here\n[{wins_A}W, {ties_A}T, {losses_A}L]",
        loc="upper left" if not lower_better else "lower right",
        frameon=True,
        prop=dict(
            color=palette[2],
            fontweight="bold",
            fontsize=13,
            ha="center",
        ),
    )
    anc_text.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    anc_text.patch.set_color("wheat")
    anc_text.patch.set_edgecolor("black")
    anc_text.patch.set_alpha(0.5)
    ax.add_artist(anc_text)

    anc_text = AnchoredText(
        f"{second_method} wins here\n[{wins_B}W, {ties_B}T, {losses_B}L]",
        loc="lower right" if not lower_better else "upper left",
        frameon=True,
        prop=dict(
            color=palette[0],
            fontweight="bold",
            fontsize=13,
            ha="center",
        ),
    )
    anc_text.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    anc_text.patch.set_color("wheat")
    anc_text.patch.set_edgecolor("black")
    anc_text.patch.set_alpha(0.5)
    ax.add_artist(anc_text)

    # Setting title if provided.
    if title is not None:
        plot.set_title(rf"{title}", fontsize=16)

    # Adding p-value if desired.
    if statistic_tests:
        if np.all(results_a == results_b):
            # raise warning
            warnings.warn(
                f"Estimators {method_a} and {method_b} have the same performance"
                "on all datasets. This may cause problems when forming cliques.",
                stacklevel=2,
            )

            p_value_t = 1
            p_value_w = 1

        else:
            from scipy.stats import ttest_rel, wilcoxon

            p_value_t = ttest_rel(
                first,
                second,
                alternative="less" if lower_better else "greater",
            )[1]

            p_value_w = wilcoxon(
                first,
                second,
                zero_method="wilcox",
                alternative="less" if lower_better else "greater",
            )[1]

        p_value_t_str = f"= {p_value_t:.3f}" if p_value_t > 1e-3 else "≤ 1e-3"
        p_value_w_str = f"= {p_value_w:.3f}" if p_value_w > 1e-3 else "≤ 1e-3"
        ttes = f"Paired t-test for equality of means: p-value {p_value_t_str}"
        wilcox = f"Wilcoxon test for equality of medians: p-value {p_value_w_str}"

        plt.figtext(
            0.5,
            0.03 if not lower_better else 0.13,
            f"{wilcox}\n{ttes}",
            fontsize=10,
            wrap=True,
            horizontalalignment="center",
            bbox=dict(
                facecolor="wheat",
                edgecolor="black",
                boxstyle="round,pad=0.5",
                alpha=0.5,
            ),
        )

    return fig, ax
