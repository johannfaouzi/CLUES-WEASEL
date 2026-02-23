# Adapted from https://github.com/MSD-IRIMAS/Multi_Comparison_Matrix

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from baycomp import SignedRankTest
from scipy.stats import wilcoxon


def capitalize_label(s):
    if len(s.split("-")) == 1:
        return s.capitalize()
    return "-".join(ss.capitalize() for ss in s.split("-"))


def get_keys_for_two_comparates(a, b):
    return f"{a}-vs-{b}"


def decode_results_data_frame(df, analysis):
    """

    Decode the necessary information from the data frame and put them into the json analysis file.

    Parameters
    ----------

    df          : pandas DataFrame containing the statistics of each comparate on multiple datasets.
        shape = (n_datasets, n_comparates), columns = [list of comparates names]
    analysis    : python dictionary

    """

    df_columns = list(df.columns)  # extract columns from data frame

    # check if dataset column name is correct

    if analysis["dataset-column"] is not None:
        if analysis["dataset-column"] not in df_columns:
            raise KeyError("The column " + analysis["dataset-column"] + " is missing.")

    # get number of examples (datasets)
    # n_datasets = len(np.unique(np.asarray(df[analysis['dataset-column']])))
    n_datasets = len(df.index)

    analysis["n-datasets"] = n_datasets  # add number of examples to dictionary

    if analysis["dataset-column"] is not None:
        analysis["dataset-names"] = list(
            df[analysis["dataset-column"]]
        )  # add example names to dict
        df_columns.remove(
            analysis["dataset-column"]
        )  # drop the dataset column name from columns list
        # and keep comparate names

    comparate_names = df_columns.copy()
    n_comparates = len(comparate_names)

    # add the information about comparates to dict
    analysis["comparate-names"] = comparate_names
    analysis["n-comparates"] = n_comparates


def get_pairwise_content(
    x,
    y,
    order_WinTieLoss="higher",
    includeProbaWinTieLoss=False,
    include_pvalue=True,
    pvalue_test="wilcoxon",
    pvalue_threshhold=0.05,
    use_mean="mean-difference",
    bayesian_rope=0.01,
):
    """

    Get the pairwise comparison between two comparates on all the given datasets.

    Parameters:
    -----------

    x                           : ndarray of shape = (n_datasets,) containing the statistics of a comparate x
        on all the datasets
    y                           : ndarray of shape = (n_datasets,) containing the statistics of a comparate y
        on all the datasets

    order_WinTieLoss            : str, default = 'higher', the order on considering a win or a loss
                       for a given statistics
    includeProbaWinTieLoss      : bool, default = False, condition whether or not include
                             the bayesian test of [1] for a probabilistic win tie loss count
    include_pvalue              : bool, default = True, condition whether or not include a pvalue stats
    pvalue_test                 : str, default = 'wilcoxon', the statistical test to produce the pvalue stats.
    pvalue_correction           : str, default = None, which correction to use for the pvalue significant test
    pvalue_threshhold           : float, default = 0.05, threshold for considering a comparison is significant
                        or not. If pvalue < pvalue_threshhold -> comparison is significant.
    use_mean                    : str, default = 'mean-difference', the mean used to comapre two comparates.
    bayesian_rope               : float, default = 0.01, the rope used in case include_ProbaWinTieLoss is True

    Returns
    -------
    content                     : python dictionary

    """

    content = {}

    if order_WinTieLoss == "lower":
        win = len(x[x < y])
        loss = len(x[x > y])
        tie = len(x[x == y])

    else:  # by default we assume higher is better
        win = len(x[x > y])
        loss = len(x[x < y])
        tie = len(x[x == y])

    content["win"] = win
    content["tie"] = tie
    content["loss"] = loss

    if include_pvalue:
        if pvalue_test == "wilcoxon":
            pvalue = wilcoxon(x=x, y=y, zero_method="pratt")[1]
            content["pvalue"] = pvalue

            if pvalue_test == "wilcoxon":
                if pvalue < pvalue_threshhold:
                    content["is-significant"] = True
                else:
                    content["is-significant"] = False

            else:
                print(f"{pvalue_test} test is not supported yet")

    if includeProbaWinTieLoss:
        bayesian_test = SignedRankTest(x=x, y=y, rope=bayesian_rope)

        p_x_wins, p_rope, p_y_wins = bayesian_test.probs()

        content["p-x-wins"] = p_x_wins
        content["p-y-wins"] = p_y_wins
        content["p-rope"] = p_rope

    if use_mean == "mean-difference":
        content["mean"] = np.mean(x) - np.mean(y)

    return content


def re_order_comparates(df_results, analysis):
    """

    Re order comparates given a specific order stats.

    Parameters
    ----------

    df_results              : pandas DataFrame containing the results of each comparate for all datasets
    analysis                : python dictionary containing the information of the pairwise comparison,
        the ordering information will be added to this dictionary

    """

    stats = []

    if analysis["order-stats"] == "average-statistic":
        for i in range(analysis["n-comparates"]):
            stats.append(analysis["average-statistic"][analysis["comparate-names"][i]])

    elif analysis["order-stats"] == "average-rank":
        if analysis["dataset-column"] is not None:
            np_results = np.asarray(
                df_results.drop([analysis["dataset-column"]], axis=1)
            )
        else:
            np_results = np.asarray(df_results)

        df = pd.DataFrame(columns=["comparate-name", "values"])

        for i, comparate_name in enumerate(analysis["comparate-names"]):
            for j in range(analysis["n-datasets"]):
                df = df.append(
                    {"comparate-name": comparate_name, "values": np_results[j][i]},
                    ignore_index=True,
                )

        rank_values = np.array(df["values"]).reshape(
            analysis["n-comparates"], analysis["n-datasets"]
        )
        df_ranks = pd.DataFrame(data=rank_values)

        average_ranks = df_ranks.rank(ascending=False).mean(axis=1)

        stats = np.asarray(average_ranks)

    elif analysis["order-stats"] == "max-wins":
        for i in range(analysis["n-comparates"]):
            wins = []

            for j in range(analysis["n-comparates"]):
                if i != j:
                    wins.append(
                        analysis[
                            analysis["comparate-names"][i]
                            + "-vs-"
                            + analysis["comparate-names"][j]
                        ]["win"]
                    )

            stats.append(int(np.max(wins)))

    elif analysis["order-stats"] == "amean-amean":
        for i in range(analysis["n-comparates"]):
            ameans = []

            for j in range(analysis["n-comparates"]):
                if i != j:
                    ameans.append(
                        analysis[
                            analysis["comparate-names"][i]
                            + "-vs-"
                            + analysis["comparate-names"][j]
                        ]["mean"]
                    )

            stats.append(np.mean(ameans))

    elif analysis["order-stats"] == "pvalue":
        for i in range(analysis["n-comparates"]):
            pvalues = []

            for j in range(analysis["n-comparates"]):
                if i != j:
                    pvalues.append(
                        analysis[
                            analysis["comparate-names"][i]
                            + "-vs-"
                            + analysis["comparate-names"][j]
                        ]["pvalue"]
                    )

            stats.append(np.mean(pvalues))

    if analysis["order-better"] == "increasing":
        ordered_indices = np.argsort(stats)
    else:  # decreasing
        ordered_indices = np.argsort(stats)[::-1]

    analysis["ordered-stats"] = list(np.asarray(stats)[ordered_indices])
    analysis["ordered-comparate-names"] = list(
        np.asarray(analysis["comparate-names"])[ordered_indices]
    )


def holms_correction(analysis):
    """
    Apply the holm correction on the pvalues of the analysis

    Parameters
    ----------

    analysis : python dictionary

    """
    pvalues = []

    for i in range(analysis["n-comparates"]):
        comparate_i = analysis["comparate-names"][i]

        for j in range(i + 1, analysis["n-comparates"]):
            if i != j:
                comparate_j = analysis["comparate-names"][j]
                pairwise_key = get_keys_for_two_comparates(comparate_i, comparate_j)
                pvalues.append(analysis[pairwise_key]["pvalue"])

    pvalues_sorted = np.sort(pvalues)

    k = 0
    m = len(pvalues)

    pvalue_times_used = {}

    for pvalue in pvalues:
        pvalue_times_used[pvalue] = 0

    for i in range(analysis["n-comparates"]):
        comparate_i = analysis["comparate-names"][i]

        for j in range(i + 1, analysis["n-comparates"]):
            if i != j:
                comparate_j = analysis["comparate-names"][j]
                pairwise_key = get_keys_for_two_comparates(comparate_i, comparate_j)
                pvalue = analysis[pairwise_key]["pvalue"]
                index_pvalue = np.where(pvalues_sorted == pvalue)[0]

                if len(index_pvalue) == 1:
                    index_pvalue = index_pvalue[0]
                else:
                    index_pvalue = index_pvalue[pvalue_times_used[pvalue]]
                    pvalue_times_used[pvalue] += 1

                pvalue_threshhold_corrected = analysis["pvalue-threshold"] / (
                    m - index_pvalue
                )

                if pvalue < pvalue_threshhold_corrected:
                    analysis[pairwise_key]["is-significant"] = True
                else:
                    analysis[pairwise_key]["is-significant"] = False

                k = k + 1

    for i in range(analysis["n-comparates"]):
        comparate_i = analysis["comparate-names"][i]

        for j in range(i + 1, analysis["n-comparates"]):
            comparate_j = analysis["comparate-names"][j]

            pairwise_key_ij = get_keys_for_two_comparates(comparate_i, comparate_j)
            pairwise_key_ji = get_keys_for_two_comparates(comparate_j, comparate_i)

            analysis[pairwise_key_ji]["is-significant"] = analysis[pairwise_key_ij][
                "is-significant"
            ]
            analysis[pairwise_key_ji]["pvalue"] = analysis[pairwise_key_ij]["pvalue"]


def get_cell_legend(
    analysis,
    win_label="r>c",
    tie_label="r=c",
    loss_label="r<c",
):
    """
    Get the content of a cell

    Parameters
    ----------

    analysis            : python dictionary
    win_label           : str, default = "r>c", the winning label to be set on the MCM
    tie_label           : str, default = "r=c", the tie label to be set on the MCM
    loss_label          : str, default = "r<c", the loss label to be set on the MCM

    Returns
    -------

    cell_legend         : str, the content of the cell
    longest_string      : int, the lenght of the longest string in the cell

    """
    cell_legend = capitalize_label(analysis["use-mean"])
    longest_string = len(cell_legend)

    win_tie_loss_string = f"{win_label} / {tie_label} / {loss_label}"
    longest_string = max(longest_string, len(win_tie_loss_string))

    cell_legend = f"{cell_legend}\n{win_tie_loss_string}"

    if analysis["include-pvalue"]:
        longest_string = max(
            longest_string, len(capitalize_label(analysis["pvalue-test"]))
        )
        pvalue_test = capitalize_label(analysis["pvalue-test"]) + " p-value"
        cell_legend = f"{cell_legend}\n{pvalue_test}"

    return cell_legend, longest_string


def get_fig_size(
    fig_size,
    n_rows,
    n_cols,
    n_info_per_cell=None,
    longest_string=None,
):
    """

    Generate figure size given the input parameters

    Parameters
    ----------

    fig_size                    : str ot tuple of two int (example : '7,10'), the height and width of the figure,
        if 'auto', use get_fig_size function in utils.py. Note that the fig size values are in
        matplotlib units
    n_rows                      : int, number of rows
    n_cols                      : int, number of columns
    n_info_per_cell             : int, default = None, the number of information placed in one cell
    longest_string              : int, default = None, the length of the longest stirng in all the cells

    """

    if isinstance(fig_size, str):
        if fig_size == "auto":
            if (n_rows == 1) and (n_cols == 2):
                size = [
                    int(max(longest_string * 0.13, 1) * n_cols),
                    int(max(n_info_per_cell * 0.1, 1) * (n_rows + 1)),
                ]

            elif n_rows <= n_cols:
                size = [
                    int(max(longest_string * 0.125, 1) * n_cols),
                    int(max(n_info_per_cell * 0.1, 1) * (n_rows + 1)),
                ]

            else:
                size = [
                    int(max(longest_string * 0.1, 1) * (n_cols + 1)),
                    int(max(n_info_per_cell * 0.125, 1) * n_rows),
                ]

            if n_rows == n_cols == 1:
                size[0] = size[0] + int(longest_string * 0.125)

            return size

        return [int(s) for s in fig_size.split(",")]

    return fig_size


def get_ticks(analysis, row_comparates, col_comparates, precision=4):
    """

    Generating tick labels for the heatmap.

    Parameters
    ----------

    analysis            : python dictionary containing all the information about the comparates and comparisons
    row_comparates      : list of str, default = None, a list of included row comparates, if None, all of
        the comparates in the study are placed in the rows.
    col_comparates      : list of str, default = None, a list of included col comparates, if None, all of
        the comparates in the study are placed in the cols.
    precision           : int, default = 4, the number of floating numbers after the decimal point

    Returns
    -------

    xticks              : list of str, containing the tick labels for each classifer
    yticks              : list of only one str, containing one tick of the comparate in question (proposed_method)

    """

    fmt = f"{precision}f"
    xticks = []
    yticks = []

    n_rows = len(row_comparates)
    n_cols = len(col_comparates)

    all_comparates = analysis["ordered-comparate-names"]
    all_stats = analysis["ordered-stats"]

    for i in range(n_rows):
        stat = all_stats[
            [
                x
                for x in range(len(all_comparates))
                if all_comparates[x] == row_comparates[i]
            ][0]
        ]

        tick_label = f"{row_comparates[i]}\n{stat:.{fmt}}"
        yticks.append(tick_label)

    for i in range(n_cols):
        stat = all_stats[
            [
                x
                for x in range(len(all_comparates))
                if all_comparates[x] == col_comparates[i]
            ][0]
        ]
        tick_label = f"{col_comparates[i]}\n{stat:.{fmt}}"
        xticks.append(tick_label)

    return xticks, yticks


def get_annotation(
    analysis,
    row_comparates,
    col_comparates,
    cell_legend,
    p_value_text,
    colormap="coolwarm",
    colorbar_value=None,
    precision=4,
):
    """
    Get the annotations of the cells

    Parameters
    ----------

    analysis                : python dictionary
    row_comparates          : list of str, default = None, a list of included row comparates, if None, all of
        the comparates in the study are placed in the rows.
    col_comparates          : list of str, default = None, a list of included col comparates, if None, all of
        the comparates in the study are placed in the cols.
    cell_legend             : str, the content of the cell
    p_value_text            : str, the content of the pvalue legend
    colormap                : str, default = 'coolwarm', the colormap used in matplotlib, if set to None,
        no color map is used and the heatmap is turned off, no colors will be seen
    colorbar_value          : str, default = 'mean-difference', the values for which the heat map colors
        are based on
    precision               : int, default = 4, the number of floating numbers after the decimal point

    Returns
    -------

    out                     : python dictionary containing data frame of annotations

    """

    fmt = f".{precision}f"

    n_rows = len(row_comparates)
    n_cols = len(col_comparates)

    pairwise_matrix = np.zeros(shape=(n_rows, n_cols))

    df_annotations = []

    n_info_per_cell = 0
    longest_string = 0

    p_value_cell_location = None
    legend_cell_location = None

    for i in range(n_rows):
        row_comparate = row_comparates[i]
        dict_to_add = {"comparates": row_comparate}
        longest_string = max(longest_string, len(row_comparate))

        for j in range(n_cols):
            col_comparate = col_comparates[j]

            if row_comparate != col_comparate:
                longest_string = max(longest_string, len(col_comparate))
                pairwise_key = get_keys_for_two_comparates(row_comparate, col_comparate)

                if colormap is not None:
                    try:
                        pairwise_matrix[i, j] = analysis[pairwise_key][colorbar_value]
                    except Exception:
                        pairwise_matrix[i, j] = analysis[pairwise_key]["mean"]

                else:
                    pairwise_matrix[i, j] = 0

                pairwise_content = analysis[pairwise_key]
                pairwise_keys = list(pairwise_content.keys())

                string_in_cell = f"{pairwise_content['mean']:{fmt}}\n"
                n_info_per_cell = 1

                if "win" in pairwise_keys:
                    string_in_cell = f"{string_in_cell}{pairwise_content['win']} / "
                    string_in_cell = f"{string_in_cell}{pairwise_content['tie']} / "
                    string_in_cell = f"{string_in_cell}{pairwise_content['loss']}\n"

                    n_info_per_cell += 1

                if "p-x-wins" in pairwise_keys:
                    string_in_cell = (
                        f"{string_in_cell}{pairwise_content['p-x-wins']:{fmt}} / "
                    )
                    string_in_cell = (
                        f"{string_in_cell}{pairwise_content['p-rope']:{fmt}} / "
                    )
                    string_in_cell = (
                        f"{string_in_cell}{pairwise_content['p-y-wins']:{fmt}}\n"
                    )

                if "pvalue" in pairwise_keys:
                    _p_value = round(pairwise_content["pvalue"], precision)
                    alpha = 10 ** (-precision)

                    if _p_value < alpha:
                        string_in_cell = rf"{string_in_cell} $\leq$ {alpha:.0e}"
                    else:
                        string_in_cell = (
                            f"{string_in_cell}{pairwise_content['pvalue']:{fmt}}"
                        )

                    n_info_per_cell += 1

                dict_to_add[col_comparate] = string_in_cell

            else:
                if legend_cell_location is None:
                    dict_to_add[row_comparate] = cell_legend
                    legend_cell_location = (i, j)
                else:
                    dict_to_add[col_comparate] = "-"
                    p_value_cell_location = (i, j)

                pairwise_matrix[i, j] = 0.0

        df_annotations.append(dict_to_add)

    if p_value_cell_location is not None:
        col_comparate = col_comparates[p_value_cell_location[1]]
        df_annotations[p_value_cell_location[0]][col_comparate] = p_value_text

    df_annotations = pd.DataFrame(df_annotations)

    out = dict(
        df_annotations=df_annotations,
        pairwise_matrix=pairwise_matrix,
        n_info_per_cell=n_info_per_cell,
        longest_string=longest_string,
        legend_cell_location=legend_cell_location,
        p_value_cell_location=p_value_cell_location,
    )

    return out


def get_limits(pairwise_matrix, can_be_negative=False, precision=4):
    """

    Get the limits, min and max, of the heatmap color bar values
    min and max produced are equal in absolute value, to insure symmetry

    Parameters
    ----------
    pairwise_matrix             : ndarray, shape = (n_comparates, n_comparates), a matrix containing
        the 1v1 statistical values (by default: difference of arithmetic mean of stats)
    can_be_negative             : bool, default = False, whether or not the values can be negative to help
            the case of the heatline
    precision                   : int, default = 4, the number of floating numbers after the decimal point


    Returns
    -------

    min_value                   : float, the min value
    max_value                   : float, the max value

    """

    if pairwise_matrix.shape[0] == 1:
        min_value = round(np.min(pairwise_matrix), precision)
        max_value = round(np.max(pairwise_matrix), precision)

        if min_value >= 0 and max_value >= 0 and (not can_be_negative):
            return min_value, max_value

        return -max(abs(min_value), abs(max_value)), max(abs(min_value), abs(max_value))

    min_value = np.min(pairwise_matrix)
    max_value = np.max(pairwise_matrix)

    if min_value < 0 or max_value < 0:
        max_min_max = max(abs(min_value), abs(max_value))

        min_value = np.sign(min_value).item() * max_min_max
        max_value = np.sign(max_value).item() * max_min_max

    return round(min_value, precision), round(max_value, precision)


def compare(
    df_results,
    output_dir="./",
    pdf_savename=None,
    png_savename=None,
    csv_savename=None,
    tex_savename=None,
    used_statistic="Accuracy",
    order_WinTieLoss="higher",
    include_ProbaWinTieLoss=False,
    bayesian_rope=0.01,
    include_pvalue=True,
    pvalue_test="wilcoxon",
    pvalue_correction=None,
    pvalue_threshold=0.05,
    use_mean="mean-difference",
    order_stats="average-statistic",
    order_better="decreasing",
    dataset_column=None,
    precision=4,
    row_comparates=None,
    col_comparates=None,
    excluded_row_comparates=None,
    excluded_col_comparates=None,
    colormap="coolwarm",
    fig_size="auto",
    font_size="auto",
    colorbar_orientation="vertical",
    colorbar_value=None,
    win_label="r>c",
    tie_label="r=c",
    loss_label="r<c",
    include_legend=True,
    show_symetry=True,
):
    if df_results.isna().any().any():
        warnings.warn(
            "There are missing values in the DataFrame. The matrix will be built using only the datasets "
            "for which the results are available for all the algorithms."
            )
        df_res = df_results.dropna(axis=0)
        print(f"Number of datasets used: {df_res.shape[0]}")
    else:
        df_res = df_results

    analysis = get_analysis(
        df_res,
        used_statistic=used_statistic,
        order_WinTieLoss=order_WinTieLoss,
        include_ProbaWinTieLoss=include_ProbaWinTieLoss,
        bayesian_rope=bayesian_rope,
        include_pvalue=include_pvalue,
        pvalue_test=pvalue_test,
        pvalue_correction=pvalue_correction,
        pvalue_threshhold=pvalue_threshold,
        use_mean=use_mean,
        order_stats=order_stats,
        order_better=order_better,
        dataset_column=dataset_column,
        precision=precision,
    )

    draw(
        analysis,
        pdf_savename=pdf_savename,
        png_savename=png_savename,
        tex_savename=tex_savename,
        csv_savename=csv_savename,
        output_dir=output_dir,
        row_comparates=row_comparates,
        col_comparates=col_comparates,
        excluded_row_comparates=excluded_row_comparates,
        excluded_col_comparates=excluded_col_comparates,
        precision=precision,
        colormap=colormap,
        fig_size=fig_size,
        font_size=font_size,
        colorbar_orientation=colorbar_orientation,
        colorbar_value=colorbar_value,
        win_label=win_label,
        tie_label=tie_label,
        loss_label=loss_label,
        include_legend=include_legend,
        show_symetry=show_symetry,
    )


def get_analysis(
    df_results,
    used_statistic="Score",
    order_WinTieLoss="higher",
    include_ProbaWinTieLoss=False,
    bayesian_rope=0.01,
    include_pvalue=True,
    pvalue_test="wilcoxon",
    pvalue_correction=None,
    pvalue_threshhold=0.05,
    use_mean="mean-difference",
    order_stats="average-statistic",
    order_better="decreasing",
    dataset_column=None,
    precision=4,
):
    """

    Get analysis of all the pairwise and multi comparate comparisons and store them in analysis
    python dictionary. With a boolean parameter, you can plot the 1v1 scatter results.

    Parameters
    ----------

    df_results              : pandas DataFrame, the csv file containing results
    output_dir              : str, default = './', the output directory for the results
    used_statistic          : str, default = 'Score', one can imagine using error, time, memory etc. instead
    save_as_json            : bool, default = True, whether or not to save the python analysis dict
        into a json file format
    plot_1v1_comparisons    : bool, default = True, whether or not to plot the 1v1 scatter results
    order_WinTieLoss        : str, default = 'higher', the order on considering a win or a loss
        for a given statistics
    include_ProbaWinTieLoss : bool, default = False, condition whether or not include
                             the bayesian test of [1] for a probabilistic win tie loss count
    bayesian_rope           : float, default = 0.01, the rope used in case include_ProbaWinTieLoss is True
    include_pvalue          : bool, default = True, condition whether or not include a pvalue stats
    pvalue_test             : str, default = 'wilcoxon', the statistical test to produce the pvalue stats.
    pvalue_correction       : str, default = None, which correction to use for the pvalue significant test
    pvalue_threshhold       : float, default = 0.05, threshold for considering a comparison is significant
        or not. If pvalue < pvalue_threshhold -> comparison is significant.
    use_mean               : str, default = 'mean-difference', the mean used to comapre two comparates.
    order_stats             : str, default = 'average-statistic', the way to order the used_statistic, default
        setup orders by average statistic over all datasets
    order_better            : str, default = 'decreasing', by which order to sort stats, from best to worse
    dataset_column          : str, default = 'dataset_name', the name of the datasets column in the csv file
    precision               : int, default = 4, the number of floating numbers after decimal point
    load_analysis           : bool, default = False, if the analysis json file is already created before, the
        use can choose to load it

    Returns
    -------
    analysis                : python dictionary containing all extracted comparisons

    """

    analysis = {
        "dataset-column": dataset_column,
        "use-mean": use_mean,
        "order-stats": order_stats,
        "order-better": order_better,
        "used-statistics": used_statistic,
        "order-WinTieLoss": order_WinTieLoss,
        "include-pvalue": include_pvalue,
        "pvalue-test": pvalue_test,
        "pvalue-threshold": pvalue_threshhold,
        "pvalue-correction": pvalue_correction,
    }

    decode_results_data_frame(df=df_results, analysis=analysis)

    if order_stats == "average-statistic":
        average_statistic = {}

    for i in range(analysis["n-comparates"]):
        comparate_i = analysis["comparate-names"][i]

        if order_stats == "average-statistic":
            average_statistic[comparate_i] = round(
                np.mean(df_results[comparate_i]), precision
            )

        for j in range(analysis["n-comparates"]):
            if i != j:
                comparate_j = analysis["comparate-names"][j]

                pairwise_key = get_keys_for_two_comparates(comparate_i, comparate_j)

                x = df_results[comparate_i]
                y = df_results[comparate_j]

                pairwise_content = get_pairwise_content(
                    x=x,
                    y=y,
                    order_WinTieLoss=order_WinTieLoss,
                    includeProbaWinTieLoss=include_ProbaWinTieLoss,
                    include_pvalue=include_pvalue,
                    pvalue_test=pvalue_test,
                    pvalue_threshhold=pvalue_threshhold,
                    use_mean=use_mean,
                    bayesian_rope=bayesian_rope,
                )

                analysis[pairwise_key] = pairwise_content

    if order_stats == "average-statistic":
        analysis["average-statistic"] = average_statistic

    if pvalue_correction == "Holm":
        holms_correction(analysis=analysis)

    re_order_comparates(df_results=df_results, analysis=analysis)

    return analysis


def draw(
    analysis,
    output_dir="./",
    pdf_savename=None,
    png_savename=None,
    csv_savename=None,
    tex_savename=None,
    row_comparates=None,
    col_comparates=None,
    excluded_row_comparates=None,
    excluded_col_comparates=None,
    precision=4,
    colormap="coolwarm",
    fig_size="auto",
    font_size="auto",
    colorbar_orientation="vertical",
    colorbar_value=None,
    win_label="r>c",
    tie_label="r=c",
    loss_label="r<c",
    show_symetry=True,
    include_legend=True,
):

    latex_string = "\\documentclass[a4,12pt]{article}\n"
    latex_string += "\\usepackage{colortbl}\n"
    latex_string += "\\usepackage{pgfplots}\n"
    latex_string += "\\usepackage[margin=2cm]{geometry}\n"
    latex_string += "\\pgfplotsset{compat=newest}\n"
    latex_string += "\\begin{document}\n"
    latex_string += "\\begin{table}\n"
    latex_string += "\\footnotesize\n"
    latex_string += "\\sffamily\n"
    latex_string += "\\begin{center}\n"

    if (col_comparates is not None) and (excluded_col_comparates is not None):
        print("Choose whether to include or exclude, not both!")
        return

    if (row_comparates is not None) and (excluded_row_comparates is not None):
        print("Choose whether to include or exclude, not both!")
        return

    if row_comparates is None:
        row_comparates = analysis["ordered-comparate-names"]
    else:
        # order comparates
        row_comparates = [
            x for x in analysis["ordered-comparate-names"] if x in row_comparates
        ]

    if col_comparates is None:
        col_comparates = analysis["ordered-comparate-names"]
    else:
        col_comparates = [
            x for x in analysis["ordered-comparate-names"] if x in col_comparates
        ]

    if excluded_row_comparates is not None:
        row_comparates = [
            x
            for x in analysis["ordered-comparate-names"]
            if x not in excluded_row_comparates
        ]

    if excluded_col_comparates is not None:
        col_comparates = [
            x
            for x in analysis["ordered-comparate-names"]
            if x not in excluded_col_comparates
        ]

    n_rows = len(row_comparates)
    n_cols = len(col_comparates)

    can_be_symmetrical = False

    if n_rows == n_cols == len(analysis["ordered-comparate-names"]):
        can_be_symmetrical = True

    if n_rows == n_cols == 1:
        figure_aspect = "equal"
        colormap = None

        if row_comparates[0] == col_comparates[0]:
            print(f"Row and Column comparates are the same, {row_comparates[0]}!")
            return
    else:
        figure_aspect = "auto"

    if (n_rows == 1) and (n_cols == 2):
        colorbar_orientation = "horizontal"

    elif (n_rows == 2) and (n_cols == 2):
        colorbar_orientation = "vertical"

    elif (n_rows == 2) and (n_cols == 1):
        colorbar_orientation = "vertical"

    elif n_rows <= 2:
        colorbar_orientation = "horizontal"

    if include_legend:
        cell_legend, longest_string = get_cell_legend(
            analysis, win_label=win_label, tie_label=tie_label, loss_label=loss_label
        )

        if analysis["include-pvalue"]:
            p_value_text = (
                f"If in bold, then\np-value < {analysis['pvalue-threshold']:.2f}"
            )

            if analysis["pvalue-correction"] is not None:
                correction = capitalize_label(analysis["pvalue-correction"])
                p_value_text = f"{p_value_text}\n{correction} correction"

        else:
            p_value_text = ""

        longest_string = max(longest_string, len(p_value_text))

    else:
        cell_legend = ""
        p_value_text = ""
        longest_string = len(f"{win_label} / {tie_label} / {loss_label}")

    annot_out = get_annotation(
        analysis=analysis,
        row_comparates=row_comparates,
        col_comparates=col_comparates,
        cell_legend=cell_legend,
        p_value_text=p_value_text,
        colormap=colormap,
        colorbar_value=colorbar_value,
        precision=precision,
    )

    df_annotations = annot_out["df_annotations"]
    pairwise_matrix = annot_out["pairwise_matrix"]

    n_info_per_cell = annot_out["n_info_per_cell"]

    legend_cell_location = annot_out["legend_cell_location"]
    p_value_cell_location = annot_out["p_value_cell_location"]

    longest_string = max(annot_out["longest_string"], longest_string)

    if csv_savename is not None:
        # todo: can add a argument to save or not
        df_annotations.to_csv(output_dir + f"{csv_savename}.csv", index=False)

    df_annotations.drop("comparates", inplace=True, axis=1)
    df_annotations_np = np.asarray(df_annotations)

    figsize = get_fig_size(
        fig_size=fig_size,
        n_rows=n_rows,
        n_cols=n_cols,
        n_info_per_cell=n_info_per_cell,
        longest_string=longest_string,
    )

    if font_size == "auto":
        if (n_rows <= 2) and (n_cols <= 2):
            font_size = 8
        else:
            font_size = 10

    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots(1, 1, figsize=(figsize[0], figsize[1]))
    ax.grid(False)

    _can_be_negative = False
    if colorbar_value is None or colorbar_value == "mean-difference":
        _can_be_negative = True
    min_value, max_value = get_limits(
        pairwise_matrix=pairwise_matrix, can_be_negative=_can_be_negative
    )

    if colormap is None:
        _colormap = "coolwarm"
        _vmin, _vmax = -2, 2
    else:
        _colormap = colormap
        _vmin = min_value + 0.2 * min_value
        _vmax = max_value + 0.2 * max_value

    if colorbar_value is None:
        _colorbar_value = capitalize_label("mean-difference")
    else:
        _colorbar_value = capitalize_label(colorbar_value)

    im = ax.imshow(
        pairwise_matrix, cmap=colormap, aspect=figure_aspect, vmin=_vmin, vmax=_vmax
    )

    if colormap is not None:
        if (
            (p_value_cell_location is None)
            and (legend_cell_location is None)
            and (colorbar_orientation == "horizontal")
        ):
            shrink = 0.4
        else:
            shrink = 0.5

        cbar = ax.figure.colorbar(
            im, ax=ax, shrink=shrink, orientation=colorbar_orientation
        )
        cbar.ax.tick_params(labelsize=font_size)
        cbar.set_label(label=capitalize_label(_colorbar_value), size=font_size)

    cm_norm = plt.Normalize(_vmin, _vmax)
    cm = plt.colormaps[_colormap]

    xticks, yticks = get_ticks(analysis, row_comparates, col_comparates, precision)
    ax.set_xticks(np.arange(n_cols), labels=xticks, fontsize=font_size)
    ax.set_yticks(np.arange(n_rows), labels=yticks, fontsize=font_size)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    ax.spines[:].set_visible(False)

    start_j = 0

    if analysis["order-stats"] == "average-statistic":
        ordering = "Mean-" + analysis["used-statistics"]
    else:
        ordering = analysis["order-stats"]

    latex_table = []
    latex_table.append(
        [f"{ordering}"]
        + [rf"\shortstack{{{_}}}".replace("\n", " \\\\ ") for _ in xticks]
    )

    for i in range(n_rows):
        row_comparate = row_comparates[i]

        latex_row = []

        if can_be_symmetrical and (not show_symetry):
            start_j = i

        for j in range(start_j, n_cols):
            col_comparate = col_comparates[j]

            cell_text_arguments = dict(
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=font_size,
            )

            if row_comparate == col_comparate:
                if p_value_cell_location is not None:
                    if (i == p_value_cell_location[0]) and (
                        j == p_value_cell_location[1]
                    ):
                        cell_text_arguments.update(
                            fontweight="bold", fontsize=font_size
                        )

                if legend_cell_location is not None:
                    if (i == legend_cell_location[0]) and (
                        j == legend_cell_location[1]
                    ):
                        cell_text_arguments.update(fontsize=font_size)

                im.axes.text(j, i, df_annotations_np[i, j], **cell_text_arguments)

                latex_cell = "\\rule{0em}{3ex} " + df_annotations_np[i, j].replace(
                    "\n", " \\\\ "
                )
                latex_row.append(
                    f"\\cellcolor[rgb]{{{','.join([str(round(_, 4)) for _ in cm(cm_norm(pairwise_matrix[i, j]))[:-1]])}}}\\shortstack{{{latex_cell}}}"  # noqa: E501
                )

                continue

            pairwise_key = get_keys_for_two_comparates(row_comparate, col_comparate)

            pairwise_content = analysis[pairwise_key]
            pairwise_keys = list(pairwise_content.keys())

            latex_bold = ""

            if "pvalue" in pairwise_keys:
                if analysis[pairwise_key]["is-significant"]:
                    cell_text_arguments.update(fontweight="bold")
                    latex_bold = "\\bfseries "

            im.axes.text(j, i, df_annotations_np[i, j], **cell_text_arguments)

            latex_cell = "\\rule{0em}{3ex} " + df_annotations_np[i, j].replace(
                "\n", " \\\\ "
            )
            latex_row.append(
                f"{latex_bold}\\cellcolor[rgb]{{{','.join([str(round(_, 4)) for _ in cm(cm_norm(pairwise_matrix[i, j]))[:-1]])}}}\\shortstack{{{latex_cell}}}"  # noqa: E501
            )

        if legend_cell_location is None:
            latex_cell = (
                "\\rule{0em}{3ex} " + f"{cell_legend}".replace("\n", " \\\\ ")
                if i == 0
                else "\\null"
            )
            latex_row.append(f"\\shortstack{{{latex_cell}}}")

        latex_table.append(
            [rf"\shortstack{{{yticks[i]}}}".replace("\n", " \\\\ ")] + latex_row
        )

    if n_cols == n_rows == 1:
        # special case when 1x1
        x = ax.get_position().x0 - 1
        y = ax.get_position().y1 - 1.5
    else:
        x = ax.get_position().x0 - 0.8
        y = ax.get_position().y1 - 1.5

    if False:
        im.axes.text(
            x,
            y,
            ordering,
            fontsize=font_size,
            horizontalalignment="center",
            verticalalignment="center",
        )

    if p_value_cell_location is None:
        x = 0
        y = n_rows

        if n_rows == n_cols == 1:
            y = 0.7
        elif (n_cols == 1) and (legend_cell_location is None):
            x = -0.5
        elif (n_rows == 1) and (n_cols <= 2) and (colorbar_orientation == "horizontal"):
            x = -0.5

        im.axes.text(
            x,
            y,
            p_value_text,
            fontsize=font_size,
            fontweight="bold",
            horizontalalignment="center",
            verticalalignment="center",
        )

    if legend_cell_location is None:
        x = n_cols - 1
        y = n_rows
        if n_rows == n_cols == 1:
            x = n_cols + 0.5
            y = 0

        elif (n_rows == 1) and (colorbar_orientation == "horizontal"):
            x = n_cols + 0.25
            y = 0

        elif n_cols == 1:
            x = 0.5

        im.axes.text(
            x,
            y,
            cell_legend,
            fontsize=font_size,
            horizontalalignment="center",
            verticalalignment="center",
        )

    if pdf_savename is not None:
        plt.savefig(
            os.path.join(output_dir + f"{pdf_savename}.pdf"), bbox_inches="tight"
        )

    if png_savename is not None:
        plt.savefig(
            os.path.join(output_dir + f"{png_savename}.png"), bbox_inches="tight"
        )

    if tex_savename is not None:
        latex_string += (
            f"\\begin{{tabular}}{{{'c' * (len(latex_table[0]) + 1)}}}\n"  # +1 for labels
        )
        for latex_row in latex_table:
            latex_string += " & ".join(latex_row) + " \\\\[1ex]" + "\n"

        if colorbar_orientation == "horizontal":
            latex_string += "\\end{tabular}\\\\\n"
        else:
            latex_string += "\\end{tabular}\n"

        latex_colorbar_0 = "\\begin{tikzpicture}[baseline=(current bounding box.center)]\\begin{axis}[hide axis,scale only axis,"  # noqa: E501
        latex_colorbar_1 = f"colormap={{cm}}{{rgb255(1)=({','.join([str(int(_ * 255)) for _ in cm(cm_norm(min_value))[:-1]])}) rgb255(2)=(220,220,220) rgb255(3)=({','.join([str(int(_ * 255)) for _ in cm(cm_norm(max_value))[:-1]])})}},"  # noqa: E501
        latex_colorbar_2 = (
            f"colorbar horizontal,point meta min={_vmin:.02f},point meta max={_vmax:.02f},"
        )
        latex_colorbar_3 = "colorbar/width=1.0em"
        latex_colorbar_4 = "}] \\addplot[draw=none] {0};\\end{axis}\\end{tikzpicture}"

        if colorbar_orientation == "horizontal":
            latex_string += (
                latex_colorbar_0
                + r"width=0sp,height=0sp,colorbar horizontal,colorbar style={width=0.25\linewidth,"
                + latex_colorbar_1
                + latex_colorbar_2
                + latex_colorbar_3
                + ",scaled x ticks=false,xticklabel style={/pgf/number format/fixed,/pgf/number format/precision=3},"
                + f"xlabel={{{_colorbar_value}}},"
                + latex_colorbar_4
            )
        else:
            latex_string += (
                latex_colorbar_0
                + r"width=1pt,colorbar right,colorbar style={height=0.25\linewidth,"
                + latex_colorbar_1
                + latex_colorbar_2
                + latex_colorbar_3
                + ",scaled y ticks=false,ylabel style={rotate=180},yticklabel style={/pgf/number format/fixed,/pgf/number format/precision=3},"  # noqa: E501
                + f"ylabel={{{_colorbar_value}}},"
                + latex_colorbar_4
            )

        latex_string += "\\end{center}\n"
        latex_string += (
            "\\caption{[...] \\textbf{"
            + f"{p_value_text}".replace("\n", " ")
            + "} [...]}\n"
        )
        latex_string += "\\end{table}\n"
        latex_string += "\\end{document}\n"

        latex_string = latex_string.replace(">", "$>$")
        latex_string = latex_string.replace("<", "$<$")

        with open(
            f"{output_dir}/{tex_savename}.tex", "w", encoding="utf8", newline="\n"
        ) as file:
            file.writelines(latex_string)

    if tex_savename is None and pdf_savename is None and png_savename is None:
        plt.show()

    plt.cla()
    plt.clf()
    plt.close()
