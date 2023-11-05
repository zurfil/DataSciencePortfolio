import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import MinMaxScaler


def _load_dataset(dataset):
    """
    Load a dataset from a given CSV file and strip it off its header.

    Parameters:
    ----------
    dataset : str
        Name (without .csv) of the dataset stored in "/data" directory without.

    Returns:
    -------
    pandas.DataFrame
        A DataFrame containing the loaded dataset.
    """
    full_data = pd.read_csv(dataset, header=None, skiprows=1)
    return full_data


def _get_datapoints(full_data):
    """
    Extract data points (features) from the provided dataset. Then scale them using MinMaxScaler.

    Parameters:
    ----------
    full_data : pandas.DataFrame
        A DataFrame containing the dataset with the last column being labels.

    Returns:
    -------
    pandas.DataFrame
        A DataFrame containing only the data points (all columns except the last one).
    """
    data = full_data.iloc[:, :-1]
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    return data


def _get_labels(full_data):
    """
    Extract labels from the provided dataset.

    Parameters:
    ----------
    full_data : pandas.DataFrame
        A DataFrame containing the dataset with the last column being labels.

    Returns:
    -------
    pandas.Series
        A Series containing the labels extracted from the last column of the dataset.
    """
    labels = full_data.iloc[:, -1]
    return labels


def _get_inject_selection(data, level):
    """
    Selects data points for noise injection.

    Parameters:
    -----------
    data : pandas.DataFrame
        The original dataset to which noise will be added.
    level : float
        The fraction of the data to which noise will be added.

    Returns:
    --------
    dict
        Selection of data points used for noise injection. Stored as row and column position.
    """
    selection_data = data.copy()
    inject_size = round(np.prod(selection_data.shape) * level)

    inject_selection = {}
    for i in range(inject_size):
        while True:
            # Repeat until a unique data point is found
            row = np.random.choice(selection_data.shape[0], 1)[0]
            column = np.random.choice(selection_data.shape[1], 1)[0]

            if [row, column] not in inject_selection.values():
                # If found, add it to selected data points
                inject_selection[i] = [row, column]
                break

    return inject_selection


def _inject_noise_gaussian(data, level):
    """
    Adds Gaussian noise to a subset of the data.

    Parameters:
    -----------
    data : pandas.DataFrame
        The original dataset to which noise will be added.
    level : float
        The fraction of the data to which noise will be added.

    Returns:
    --------
    pandas.DataFrame
        The dataset with Gaussian noise added to a subset of its entries.
    """
    data_copy = data.copy()
    gauss_data = pd.DataFrame(data=data_copy)
    inject_selection = _get_inject_selection(gauss_data, level)

    for index, coordinates in inject_selection.items():
        # coordinates[0] -> row, coordinates[1] -> column
        mean = gauss_data.iloc[:, coordinates[1]].mean()  # mean of a column to which data point belongs
        std_dev = gauss_data.iloc[:, coordinates[1]].std()  # std_dev of a column to which data point belongs
        gauss_data.at[coordinates[0], coordinates[1]] = np.random.normal(mean, std_dev)

    return gauss_data


def _inject_noise_uniform(data, level):
    """
    Adds Uniform noise to a subset of the data.

    Parameters:
    -----------
    data : pandas.DataFrame
        The original dataset to which noise will be added.
    level : float
        The fraction of the data to which noise will be added.

    Returns:
    --------
    pandas.DataFrame
        The dataset with Uniform noise added to a subset of its entries.
    """
    data_copy = data.copy()
    uniform_data = pd.DataFrame(data=data_copy)

    inject_selection = _get_inject_selection(uniform_data, level)

    for index, coordinates in inject_selection.items():
        # coordinates[0] -> row, coordinates[1] -> column
        low = uniform_data.iloc[:, coordinates[1]].min()  # min value of a column to which data point belongs
        high = uniform_data.iloc[:, coordinates[1]].max()  # max value of a column to which data point belongs
        uniform_data.at[coordinates[0], coordinates[1]] = np.random.uniform(low, high)

    return uniform_data


def _inject_noise_quantizaion(data, level):
    """
    Adds Quantization noise to a subset of the data.

    Parameters:
    -----------
    data : pandas.DataFrame
        The original dataset to which noise will be added.
    level : float
        The fraction of the data to which noise will be added.

    Returns:
    --------
    pandas.DataFrame
        The dataset with Quantization noise added to a subset of its entries.
    """
    data_copy = data.copy()
    quant_data = pd.DataFrame(data=data_copy)
    inject_selection = _get_inject_selection(quant_data, level)

    for index, coordinates in inject_selection.items():
        quant_data.at[coordinates[0], coordinates[1]] = round(quant_data.at[coordinates[0], coordinates[1]] * 5) / 5

    return quant_data


def get_baselines(datasets, noise_levels):
    noise_functions = {
        "Gaussian": _inject_noise_gaussian,
        "Uniform": _inject_noise_uniform,
        "Quantization": _inject_noise_quantizaion
    }

    header_list = ["Dataset",
                   "Noise Type",
                   "Noise Level",
                   "Normalized Mutual Information"]

    output_df = pd.DataFrame(columns=header_list)

    for dataset in datasets:
        # Load dataset
        full_data = _load_dataset(f"data/{dataset}.csv")
        # Isolate data
        data = _get_datapoints(full_data)
        # Isolate labels
        labels = _get_labels(full_data)
        # Calculate how many final clusters
        final_n_clusters = labels.nunique()
        for noise_name, noise_function in noise_functions.items():
            for noise_level in noise_levels:
                noise_data = noise_function(data, noise_level)
                scores = [
                    normalized_mutual_info_score(
                        KMeans(n_clusters=final_n_clusters, n_init="auto").fit_predict(noise_data), labels)
                    for _ in range(5)]
                new_row = {"Dataset": dataset,
                           "Noise Type": noise_name,
                           "Noise Level": noise_level,
                           "Normalized Mutual Information": np.mean(scores)}
                output_df.loc[len(output_df)] = new_row

    output_df.to_csv(f"results/baselines.csv", sep=',', index=False)
