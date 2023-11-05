
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, HDBSCAN, Birch, MiniBatchKMeans
from sklearn.metrics import (silhouette_score,
                             davies_bouldin_score,
                             calinski_harabasz_score,
                             adjusted_rand_score,
                             normalized_mutual_info_score)
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


def _load_dataset_with_header(dataset):
    """
    Load a dataset from a given CSV file.

    Parameters:
    ----------
    dataset : str
        Name (without .csv) of the dataset stored in "/data" directory without.

    Returns:
    -------
    pandas.DataFrame
        A DataFrame containing the loaded dataset.
    """
    full_data = pd.read_csv(dataset)
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
    numpy.array
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
    numpy.array
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
    numpy.array
        The dataset with Quantization noise added to a subset of its entries.
    """
    data_copy = data.copy()
    quant_data = pd.DataFrame(data=data_copy)
    inject_selection = _get_inject_selection(quant_data, level)

    for index, coordinates in inject_selection.items():
        quant_data.at[coordinates[0], coordinates[1]] = round(quant_data.at[coordinates[0], coordinates[1]] * 5) / 5

    return quant_data


def _make_kmeans_pool(data, n_algorithms=50):
    """
    Create a pool of KMeans clustering results with varying cluster numbers.

    Parameters:
    -----------
    data : pandas.DataFrame
        The original dataset to which noise will be added.
    n_algorithms : int, optional
        Number of KMeans algorithms to run. Default is 50.

    Returns:
    --------
    list
        List of clustering results with varying cluster numbers.
    """
    kmeans_list = []
    n_samples = len(data)
    max_random = int(np.sqrt(n_samples))

    for i in range(n_algorithms):
        kmeans_member = (MiniBatchKMeans(n_clusters=np.random.randint(2, max_random), n_init=10).fit_predict(data))
        kmeans_list.append(kmeans_member)

    return kmeans_list


def _make_agglomeration_pool(data, n_algorithms=50):
    """
    Create a pool of Agglomerative clustering results with varying cluster numbers and linkage methods.

    Parameters:
    -----------
    data : pandas.DataFrame
        The original dataset to which noise will be added.
    n_algorithms : int, optional
        Number of Agglomerative algorithms to run. Default is 50.

    Returns:
    --------
    list
        List of clustering results with varying cluster numbers and linkage methods.
    """
    agglomeration_list = []
    linkage = ["single", "ward", "complete"]
    n_samples = len(data)
    max_random = int(np.sqrt(n_samples))

    for _ in range(n_algorithms):
        agglomeration_member = (AgglomerativeClustering(n_clusters=np.random.randint(2, max_random),
                                                        linkage=np.random.choice(linkage)).fit_predict(data))
        agglomeration_list.append(agglomeration_member)

    return agglomeration_list


def _make_multimember_pool(data, n_algorithms=50):
    """
    Create a pool of clustering results from a set of different clustering algorithms.

    Parameters:
    -----------
    data : pandas.DataFrame
        The original dataset to which noise will be added.
    n_algorithms : int, optional
        Number of algorithms to run. Default is 50.

    Returns:
    --------
    list
        List of clustering results with varying cluster numbers and linkage methods.
    """
    multimember_list = []
    linkage = ["single", "ward", "complete"]
    algorithm = ["balltree", "kdtree", "brute"]  # HDBSCAN algorithm parameters
    method = ["leaf", "eom"]  # HDBSCAN cluster selection methods
    n_samples = len(data)
    max_random = int(np.sqrt(n_samples))
    max_cluster_size = round(n_samples*0.25)

    for _ in range(n_algorithms):
        if len(multimember_list) >= n_algorithms:
            break

        kmeans_member = MiniBatchKMeans(n_clusters=np.random.randint(2, max_random),
                                        n_init=10).fit_predict(data)
        multimember_list.append(kmeans_member)

        agglomeration_member = AgglomerativeClustering(n_clusters=np.random.randint(2, max_random),
                                                       linkage=np.random.choice(linkage)).fit_predict(data)
        multimember_list.append(agglomeration_member)

        hdbscan_member = HDBSCAN(cluster_selection_method=np.random.choice(method),
                                 algorithm=np.random.choice(algorithm),
                                 min_cluster_size=2,
                                 max_cluster_size=max_cluster_size, n_jobs=-1).fit_predict(data)
        multimember_list.append(hdbscan_member)

    return multimember_list


def _select_best_members(data, clusters_list, target_n_members=25):
    """
    Selects the most different algorithm compared to the algorithms in the target pool based on the mean NMI score
    of the algorithm against all the algorithms in the target pool. Moves the one with the lowest NMI score
    to the target pool.

    Parameters:
    -----------
    data : array-like or pandas DataFrame
        The dataset for which silhouette scores are being calculated.
    clusters_list : list
        List containing the cluster labels for the dataset.
    target_n_members: int, optional
        Desired number of target pool members. Default is 25.

    Returns:
    --------
    list
        List of the most dissimilar clusters.
    """
    all_members = np.array(clusters_list)
    initial_scores = np.array([silhouette_score(data, cluster) for cluster in clusters_list])
    best_member = np.argmax(initial_scores)

    # Assign the best member to the target pool
    target_pool = [best_member]

    # Create a member_pool with all indexes but the best one
    member_pool = [i for i in range(len(clusters_list)) if i != best_member]

    while len(target_pool) < target_n_members:
        average_nmi_scores = [np.mean([normalized_mutual_info_score(all_members[i], all_members[j])
                                       for j in target_pool])for i in member_pool]

        lowest_nmi = member_pool[np.argmin(average_nmi_scores)]

        # Add the lowest one to the target pool and remove it from member pool
        target_pool.append(lowest_nmi)
        member_pool.remove(lowest_nmi)

    # Return a list of lists of prediction labels where each prediction algorithm is a member of the target_pool
    return all_members[target_pool].tolist()


def _make_similarity_matrix(labels_list):
    """
    Calculate a similarity matrix based on a list of cluster labels.

    Parameters:
    -----------
    labels_list : list of array-like
        List containing clustering labels for each method.

    Returns:
    --------
    numpy.ndarray
        Normalized similarity matrix where each entry is between 0 and 1.
    """
    n_samples = len(labels_list[0])

    # Create an empty array to hold the cumulative similarity
    cumulative_similarity = np.zeros((n_samples, n_samples))

    for labels in labels_list:
        labels = np.array(labels)

        # Create a boolean matrix where True indicates the labels are equal
        equal_labels = labels[:, None] == labels

        # Mask out positions where either of the labels is -1
        valid_labels = (labels[:, None] != -1) & (labels != -1)

        # Combine the conditions and convert to integer (True -> 1, False -> 0)
        cumulative_similarity += (equal_labels & valid_labels).astype(int)

    # Normalize the similarity matrix
    similarity_matrix = cumulative_similarity / len(labels_list)

    return similarity_matrix


def _cluster_matrix_results(similarity_matrix, n_clusters, linkage):
    """
    Cluster the results based on a precomputed similarity matrix.

    Parameters:
    -----------
    similarity_matrix : array-like, shape = [n_samples, n_samples]
        Precomputed similarity matrix where higher values indicate greater similarity.
    n_clusters : int
        The number of clusters to form.
    linkage : str
        The linkage method to use.

    Returns:
    --------
    numpy.ndarray
        Array of cluster labels for each sample.
    """
    final_clustering = AgglomerativeClustering(n_clusters=n_clusters,
                                               metric="precomputed",
                                               linkage=linkage.lower()).fit_predict(1 - similarity_matrix)

    return final_clustering


def _save_cluster_fig(data, data_name, noise_type, alg_name, level, i, final_clusters):
    """
    Generate and save a 2D scatter plot of clustered data.

    The function creates a 2D scatter plot based on the first two columns/features of the provided
    data. The color of the points represents the clusters to which they belong. The plot is then
    saved with a specific naming convention.

    Parameters:
    -----------
    - data : pandas.DataFrame
        Data to be plotted. Assumes it has at least two columns representing the
        features for the scatter plot.
    - data_name : str
        Name of the dataset.
    - noise_type : str
        Type of noise applied to the data.
    - alg_name : str
        Name of the clustering algorithm used.
    - level : int or float
        Level or intensity of the noise.
    - i : int
        Run number.
    - final_clusters : list or pandas.Series
        Predicted cluster labels for each data point.
    """
    data_no_true_labels = pd.DataFrame(data=data)
    data_no_true_labels["pred_labels"] = final_clusters
    plt.clf()
    plt.scatter(data_no_true_labels.iloc[:, 0], data_no_true_labels.iloc[:, 1], c=data_no_true_labels["pred_labels"])
    plt.title(f"{data_name.title()}_{noise_type}_level={level}_{alg_name}_run-{i+1}")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.savefig(f"output_graphs/{data_name}_{noise_type}_{level}_{alg_name}_{i+1}.png")


def make_test_diversity(datasets, noise_levels):
    """
    Perform clustering tests on various datasets and under different conditions.

    Performs clustering for each dataset, noise type, noise level, clustering algorithm and linkage method and
    then extracts the results into a .csv file named "results_diversity_total.csv".

    Parameters:
    ----------
    datasets : list of str
        List of dataset names (without .csv extension and directory path) to be used in the tests.
        Must be stored in "/data" directory.

    noise_levels : list of floats
         Levels of noise intensity to be injected into the datasets.
    """
    linkage_list = ["Average", "Single", "Complete"]
    algorithm_functions = {
        "K-Means": _make_kmeans_pool,
        "Agglomerative": _make_agglomeration_pool,
        "MultiMember": _make_multimember_pool
    }
    noise_functions = {
        "Gaussian": _inject_noise_gaussian,
        "Uniform": _inject_noise_uniform,
        "Quantization": _inject_noise_quantizaion
    }

    total_results_df = pd.DataFrame(columns=["Dataset",
                                             "Noise Type",
                                             "Algorithm",
                                             "Linkage",
                                             "Noise Level",
                                             "Normalized Mutual Information"])

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
                for algorithm, algorithm_function in algorithm_functions.items():
                    for linkage in linkage_list:
                        scores = []

                        # Perform the test 5 times
                        for i in range(5):
                            member_pool = algorithm_function(noise_data)
                            clustering = _select_best_members(data, member_pool)

                            matrix = _make_similarity_matrix(clustering)

                            pred_labels = _cluster_matrix_results(matrix, final_n_clusters, linkage)

                            scores.append(normalized_mutual_info_score(pred_labels, labels))

                            # OPTIONAL: Un-comment it to make graphs visualising cluster assignments for each run.
                            # Prior to it, create "output_graphs" directory in main project directory if not present.
                            # WARNING: Extreme performance decrease

                            # if data.shape[1] == 2:
                            #    _save_cluster_fig(noise_data, dataset, noise_name,
                            #                      algorithm, noise_level, i, pred_labels)

                        new_row = {
                            "Dataset": dataset,
                            "Noise Type": noise_name,
                            "Noise Level": noise_level,
                            "Algorithm": algorithm,
                            "Linkage": linkage,
                            "Normalized Mutual Information": f"{round(np.mean(scores), 4)} " +
                                                             f"({round(np.std(scores),2)})"
                        }

                        print(new_row)
                        total_results_df.loc[len(total_results_df)] = new_row

    # Log the results in a .csv file
    total_results_df.to_csv(f"results/results_diversity_total.csv", sep=',', index=False)
