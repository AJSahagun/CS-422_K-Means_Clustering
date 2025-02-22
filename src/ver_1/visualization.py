import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_clustering_analysis(X_scaled, df, cluster_labels, true_labels, cluster_centers, wcss_values, conf_matrix,
                             scaler=None):
    """
    Parameters:
    -----------
    X_scaled : array-like
        Scaled feature data
    df : pandas.DataFrame
        Original dataframe with features
    cluster_labels : array-like
        Predicted cluster labels
    true_labels : array-like
        True species labels
    cluster_centers : array-like
        Coordinates of cluster centers
    wcss_values : list
        WCSS values for different k
    conf_matrix : array-like
        Confusion matrix
    scaler : sklearn.preprocessing.StandardScaler
        Scaler used for feature scaling
    """
    plt.figure(figsize=(12, 8))

    # 1. Elbow Method Plot
    plt.subplot(2, 2, 1)
    plt.plot(range(1, len(wcss_values) + 1), wcss_values, marker='o', linewidth=2, markersize=8)
    plt.title('Elbow Method', fontsize=14, pad=15)
    plt.xlabel('Number of Clusters (K)', fontsize=12)
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # 2. Cluster Visualization
    plt.subplot(2, 2, 2)

    # Transform centroids back to original scale if scaler is provided
    if scaler is not None:
        cluster_centers = scaler.inverse_transform(cluster_centers)

    plt.scatter(df['PetalLengthCm'], df['PetalWidthCm'],
                c=cluster_labels, cmap='viridis',
                s=100, alpha=0.6)
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
                s=300, c='red', marker='X', linewidth=2,
                label='Centroids')
    plt.title('K-means Clustering Results', fontsize=14, pad=15)
    plt.xlabel('Petal Length (cm)', fontsize=12)
    plt.ylabel('Petal Width (cm)', fontsize=12)
    plt.legend(fontsize=10)

    # 3. True Labels Visualization
    plt.subplot(2, 2, 3)
    unique_species = np.unique(true_labels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_species)))

    for species, color in zip(unique_species, colors):
        mask = true_labels == species
        plt.scatter(df.loc[mask, 'PetalLengthCm'],
                    df.loc[mask, 'PetalWidthCm'],
                    label=species, color=color, s=100, alpha=0.6)

    plt.title('Actual Species Distribution', fontsize=14, pad=15)
    plt.xlabel('Petal Length (cm)', fontsize=12)
    plt.ylabel('Petal Width (cm)', fontsize=12)
    plt.legend(fontsize=10)

    # 4. Confusion Matrix
    plt.subplot(2, 2, 4)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=unique_species,
                yticklabels=unique_species)
    plt.title('Confusion Matrix', fontsize=14, pad=15)
    plt.xlabel('Predicted Species', fontsize=12)
    plt.ylabel('True Species', fontsize=12)

    plt.tight_layout(h_pad=0.5, w_pad=0.5)
    plt.show()


def calculate_wcss(X_scaled, kmeans_class, max_k=10):
    """
    Parameters:
    -----------
    X_scaled : array-like
        Scaled feature data
    kmeans_class : class
        KMeans class to use (either custom or sklearn)
    max_k : int
        Maximum number of clusters to try

    Returns:
    --------
    list : WCSS values for each k
    """
    wcss = []
    for k in range(1, max_k + 1):
        kmeans = kmeans_class(n_clusters=k)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.get_wcss())
    return wcss


def find_elbow_point(wcss_values):
    """
    Parameters:
        wcss_values: A list of WCSS values corresponding to different K values.

    Returns:
        The estimated optimal K value (index + 1), or None if no clear elbow is found.
    """

    if len(wcss_values) < 3:
        return len(wcss_values)

    diffs = []
    for i in range(1, len(wcss_values)):
        diffs.append(wcss_values[i - 1] - wcss_values[i])

    second_diffs = []
    for i in range(1, len(diffs)):
        second_diffs.append(diffs[i - 1] - diffs[i])

    # Find the point where the second derivative changes sign significantly.
    # This indicates a change in the rate of decrease.

    max_second_diff_index = 0
    max_second_diff_value = 0

    for i, value in enumerate(second_diffs):
        if i == 0:
            max_second_diff_value = abs(value)
        elif abs(value) > max_second_diff_value:
            max_second_diff_value = abs(value)
            max_second_diff_index = i

    elbow_index = max_second_diff_index +1 #add one to account for the first set of differences.
    optimal_k = elbow_index + 1 #add one to account for the fact that index 0 is K=1.

    return optimal_k