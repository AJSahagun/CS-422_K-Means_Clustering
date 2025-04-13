import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.colors import to_rgba


def plot_clustering_analysis(X_scaled, df, cluster_labels, true_labels, cluster_centers, wcss_values, conf_matrix,
                             scaler=None):
    """
    Parameters:
    -----------
    X_scaled : array-like
        Scaled feature data.
    df : pandas.DataFrame
        Original dataframe containing the features.
    cluster_labels : array-like
        Predicted cluster labels.
    true_labels : array-like
        True species labels.
    cluster_centers : array-like
        Coordinates of cluster centers.
    wcss_values : list
        WCSS values for different k.
    conf_matrix : array-like
        Confusion matrix.
    scaler : sklearn.preprocessing.StandardScaler, optional
        Scaler used for feature scaling.
    """
    fig = plt.figure(figsize=(10, 8))

    # 1. Elbow Method Plot (2D)
    ax1 = fig.add_subplot(2, 2, 2)
    ax1.plot(range(1, len(wcss_values) + 1), wcss_values, marker='o', linewidth=2, markersize=8)
    ax1.set_title('Elbow Method', fontsize=14, pad=15)
    ax1.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax1.set_ylabel('Within-Cluster Sum of Squares (WCSS)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # 2. Cluster Visualization (3D)
    ax2 = fig.add_subplot(2, 2, 3, projection='3d')
    # If a scaler is provided, inverse-transform the cluster centers to the original scale
    if scaler is not None:
        cluster_centers = scaler.inverse_transform(cluster_centers)

    # 3D scatter for all points colored by cluster labels
    ax2.scatter(df['SepalLengthCm'], df['SepalWidthCm'], df['PetalLengthCm'],
                c=cluster_labels, cmap='viridis', s=100, alpha=0.6)

    # Plot cluster centers
    ax2.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2],
                s=300, c='red', marker='X', linewidth=2, label='Centroids')

    ax2.set_title('K-means Clustering Results', fontsize=14, pad=15)
    ax2.set_xlabel('Sepal Length (cm)', fontsize=12)
    ax2.set_ylabel('Sepal Width (cm)', fontsize=12)
    ax2.set_zlabel('Petal Length (cm)', fontsize=12)
    ax2.legend(fontsize=10)

    # 3. True Labels Visualization (3D)
    ax3 = fig.add_subplot(2, 2, 1, projection='3d')
    unique_species = np.unique(true_labels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_species)))

    for species, color in zip(unique_species, colors):
        mask = true_labels == species
        ax3.scatter(df.loc[mask, 'SepalLengthCm'],
                    df.loc[mask, 'SepalWidthCm'],
                    df.loc[mask, 'PetalLengthCm'],
                    label=species, color=color, s=100, alpha=0.6)

    ax3.set_title('Actual Species Distribution', fontsize=14, pad=15)
    ax3.set_xlabel('Sepal Length (cm)', fontsize=12)
    ax3.set_ylabel('Sepal Width (cm)', fontsize=12)
    ax3.set_zlabel('Petal Length (cm)', fontsize=12)
    ax3.legend(fontsize=10)

    # 4. Confusion Matrix (2D)
    ax4 = fig.add_subplot(2, 2, 4)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=unique_species, yticklabels=unique_species, ax=ax4)
    ax4.set_title('Confusion Matrix', fontsize=14, pad=15)
    ax4.set_xlabel('Predicted Species', fontsize=12)
    ax4.set_ylabel('True Species', fontsize=12)

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

    elbow_index = max_second_diff_index + 1  # add one to account for the first set of differences.
    optimal_k = elbow_index + 1  # add one to account for the fact that index 0 is K=1.

    return optimal_k


def plot_kmeans_iterations(X_scaled, df, centroid_history, labels_history, iterations, scaler=None):
    """
    Visualize the evolution of K-means clustering iterations.

    Parameters:
    -----------
    X_scaled : array-like
        Scaled feature data.
    df : pandas.DataFrame
        Original dataframe containing the features.
    centroid_history : list
        List of centroid arrays for each iteration.
    labels_history : list
        List of label arrays for each iteration.
    iterations : int
        Number of iterations performed.
    scaler : sklearn.preprocessing.StandardScaler, optional
        Scaler used for feature scaling.
    """
    # Create subplots (one for each iteration)
    n_iterations = min(iterations + 1, 9)  # Limit to 9 plots or actual iterations
    rows = (n_iterations + 2) // 3  # Ceiling division
    fig = plt.figure(figsize=(10, 8 * rows))
    fig.tight_layout()

    # If scaler provided, transform centroids back to original scale
    if scaler is not None:
        centroid_history_orig = [scaler.inverse_transform(centroids) for centroids in centroid_history]
    else:
        centroid_history_orig = centroid_history

    # Generate a consistent color map for clusters across all iterations
    num_clusters = centroid_history[0].shape[0]
    cluster_colors = plt.cm.viridis(np.linspace(0, 1, num_clusters))

    # Create plots for each iteration
    for i in range(min(iterations, 8)):
        ax = fig.add_subplot(rows, 3, i + 1, projection='3d')

        # Plot data points with cluster labels
        if i < len(labels_history):
            # Create a scatter plot for each cluster with consistent colors
            for cluster_idx in range(num_clusters):
                mask = labels_history[i] == cluster_idx
                ax.scatter(
                    df.loc[mask, 'SepalLengthCm'],
                    df.loc[mask, 'SepalWidthCm'],
                    df.loc[mask, 'PetalLengthCm'],
                    color=cluster_colors[cluster_idx],
                    s=80,
                    alpha=0.6,
                    label=f'Cluster {cluster_idx + 1}'
                )

        # Plot centroids
        if i < len(centroid_history_orig):
            # Plot each centroid with a color matching its cluster
            for cluster_idx, centroid in enumerate(centroid_history_orig[i]):
                ax.scatter(
                    centroid[0],
                    centroid[1],
                    centroid[2],
                    s=200,
                    color=cluster_colors[cluster_idx],
                    marker='X',
                    edgecolor='black',
                    linewidth=2
                )

        ax.set_title(f'Iteration {i + 1}', fontsize=12)
        ax.set_xlabel('Sepal Length (cm)', fontsize=10)
        ax.set_ylabel('Sepal Width (cm)', fontsize=10)
        ax.set_zlabel('Petal Length (cm)', fontsize=10)

        # Only show legend on the first plot to avoid clutter
        if i == 0:
            ax.legend(fontsize=8, loc='upper right')

    # Add final result
    if iterations > 0:
        ax = fig.add_subplot(rows, 3, min(iterations, 9), projection='3d')

        # Plot the final clustering result
        if len(labels_history) > 0:
            # Create a scatter plot for each cluster with consistent colors
            for cluster_idx in range(num_clusters):
                mask = labels_history[-1] == cluster_idx
                ax.scatter(
                    df.loc[mask, 'SepalLengthCm'],
                    df.loc[mask, 'SepalWidthCm'],
                    df.loc[mask, 'PetalLengthCm'],
                    color=cluster_colors[cluster_idx],
                    s=80,
                    alpha=0.6,
                    label=f'Cluster {cluster_idx + 1}'
                )

        # Plot the final centroids
        if len(centroid_history_orig) > 0:
            # Plot each centroid with a color matching its cluster
            for cluster_idx, centroid in enumerate(centroid_history_orig[-1]):
                ax.scatter(
                    centroid[0],
                    centroid[1],
                    centroid[2],
                    s=200,
                    color=cluster_colors[cluster_idx],
                    marker='X',
                    edgecolor='black',
                    linewidth=2
                )

        ax.set_title(f'Iteration {iterations+1} (Final)', fontsize=12)
        ax.set_xlabel('Sepal Length (cm)', fontsize=10)
        ax.set_ylabel('Sepal Width (cm)', fontsize=10)
        ax.set_zlabel('Petal Length (cm)', fontsize=10)
        ax.legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    plt.show()


def create_kmeans_animation(X_scaled, df, centroid_history, labels_history, scaler=None):
    """
    Create an animation of K-means iterations.

    Parameters:
    -----------
    X_scaled : array-like
        Scaled feature data.
    df : pandas.DataFrame
        Original dataframe with feature columns.
    centroid_history : list
        List of centroid arrays for each iteration.
    labels_history : list
        List of label arrays for each iteration.
    scaler : sklearn.preprocessing.StandardScaler, optional
        Scaler used for feature scaling.

    Returns:
    --------
    animation : matplotlib.animation.FuncAnimation
        Animation object that can be saved or displayed.
    """
    # If scaler provided, transform centroids back to original scale
    if scaler is not None:
        centroid_history_orig = [scaler.inverse_transform(centroids) for centroids in centroid_history]
    else:
        centroid_history_orig = centroid_history

    # Create figure and 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Setup plot limits
    ax.set_xlim([df['SepalLengthCm'].min() - 0.5, df['SepalLengthCm'].max() + 0.5])
    ax.set_ylim([df['SepalWidthCm'].min() - 0.5, df['SepalWidthCm'].max() + 0.5])
    ax.set_zlim([df['PetalLengthCm'].min() - 0.5, df['PetalLengthCm'].max() + 0.5])

    # Set labels
    ax.set_xlabel('Sepal Length (cm)')
    ax.set_ylabel('Sepal Width (cm)')
    ax.set_zlabel('Petal Length (cm)')

    # Generate consistent colors for clusters
    num_clusters = centroid_history[0].shape[0]
    cluster_colors = plt.cm.viridis(np.linspace(0, 1, num_clusters))

    # Create scatter plots for each cluster (initialized with first iteration)
    scatter_plots = []
    for cluster_idx in range(num_clusters):
        # Initialize with empty data for this cluster
        mask = labels_history[0] == cluster_idx
        scatter = ax.scatter(
            df.loc[mask, 'SepalLengthCm'],
            df.loc[mask, 'SepalWidthCm'],
            df.loc[mask, 'PetalLengthCm'],
            color=cluster_colors[cluster_idx],
            s=80,
            alpha=0.6,
            label=f'Cluster {cluster_idx + 1}'
        )
        scatter_plots.append(scatter)

    # Create centroid plots for each cluster
    centroid_plots = []
    for cluster_idx in range(num_clusters):
        centroid = ax.scatter(
            [centroid_history_orig[0][cluster_idx, 0]],
            [centroid_history_orig[0][cluster_idx, 1]],
            [centroid_history_orig[0][cluster_idx, 2]],
            s=200,
            color=cluster_colors[cluster_idx],
            marker='X',
            edgecolor='black',
            linewidth=2
        )
        centroid_plots.append(centroid)

    title = ax.set_title('K-means Iteration: 0', fontsize=14)
    ax.legend(fontsize=10)

    def update(frame):
        title.set_text(f'K-means Iteration: {frame}')

        # Update cluster assignments and centroids
        for cluster_idx in range(num_clusters):
            # Update data points for this cluster
            mask = labels_history[frame] == cluster_idx

            if np.any(mask):  # Only update if there are points in this cluster
                scatter_plots[cluster_idx]._offsets3d = (
                    df.loc[mask, 'SepalLengthCm'],
                    df.loc[mask, 'SepalWidthCm'],
                    df.loc[mask, 'PetalLengthCm']
                )
            else:
                # If no points in this cluster, show empty plot
                scatter_plots[cluster_idx]._offsets3d = ([], [], [])

            # Update centroid for this cluster
            centroid_plots[cluster_idx]._offsets3d = (
                [centroid_history_orig[frame][cluster_idx, 0]],
                [centroid_history_orig[frame][cluster_idx, 1]],
                [centroid_history_orig[frame][cluster_idx, 2]]
            )

        return scatter_plots + centroid_plots + [title]

    # Create animation
    anim = FuncAnimation(fig, update, frames=len(centroid_history),
                         interval=1000, blit=False)

    plt.tight_layout()
    return anim