import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.animation import FuncAnimation

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
        True country labels (updated from species).
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

    # 1. Elbow Method Plot (2D) - unchanged
    ax1 = fig.add_subplot(2, 2, 2)
    ax1.plot(range(1, len(wcss_values) + 1), wcss_values, marker='o', linewidth=2, markersize=8)
    ax1.set_title('Elbow Method', fontsize=14, pad=15)
    ax1.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax1.set_ylabel('Within-Cluster Sum of Squares (WCSS)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # 2. Cluster Visualization (3D) - updated features
    ax2 = fig.add_subplot(2, 2, 3, projection='3d')
    if scaler is not None:
        cluster_centers = scaler.inverse_transform(cluster_centers)

    # 3D scatter for all points colored by cluster labels - updated to Quantity, UnitPrice, CustomerID
    ax2.scatter(df['A'], df['A_Coef'], df['LKG'],
                c=cluster_labels, cmap='viridis', s=100, alpha=0.6)

    # Plot cluster centers
    ax2.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2],
                s=300, c='red', marker='X', linewidth=2, label='Centroids')

    ax2.set_title('K-means Clustering Results', fontsize=14, pad=15)
    ax2.set_xlabel('A', fontsize=12)  # Updated label
    ax2.set_ylabel('A_Coef', fontsize=12)  # Updated label
    ax2.set_zlabel('LKG', fontsize=12)  # Updated label
    ax2.legend(fontsize=10)

    # 3. True Labels Visualization (3D) - updated features and country labels
    ax3 = fig.add_subplot(2, 2, 1, projection='3d')
    unique_countries = np.unique(true_labels)  # Updated from unique_species
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_countries)))

    for country, color in zip(unique_countries, colors):
        mask = true_labels == country
        ax3.scatter(df.loc[mask, 'A'],
                    df.loc[mask, 'A_Coef'],
                    df.loc[mask, 'LKG'],
                    label=country, color=color, s=100, alpha=0.6)

    ax3.set_title('Actual Seed Distribution', fontsize=14, pad=15)  # Updated title
    ax3.set_xlabel('A', fontsize=12)  # Updated label
    ax3.set_ylabel('A_Coef', fontsize=12)  # Updated label
    ax3.set_zlabel('LKG', fontsize=12)  # Updated label
    ax3.legend(fontsize=10)

    # 4. Confusion Matrix (2D) - updated to use countries
    ax4 = fig.add_subplot(2, 2, 4)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=unique_countries, yticklabels=unique_countries, ax=ax4)
    ax4.set_title('Confusion Matrix', fontsize=14, pad=15)
    ax4.set_xlabel('Predicted Label', fontsize=12)  # Updated label
    ax4.set_ylabel('True Label', fontsize=12)  # Updated label

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

    max_second_diff_index = 0
    max_second_diff_value = 0

    for i, value in enumerate(second_diffs):
        if i == 0:
            max_second_diff_value = abs(value)
        elif abs(value) > max_second_diff_value:
            max_second_diff_value = abs(value)
            max_second_diff_index = i

    elbow_index = max_second_diff_index + 1
    optimal_k = elbow_index + 2

    return optimal_k

def plot_kmeans_iterations(X_scaled, df, centroid_history, labels_history, iterations, scaler=None):
    """
    Visualize the evolution of K-means clustering iterations - updated features
    """
    n_iterations = min(iterations + 1, 9)
    rows = (n_iterations + 2) // 3
    fig = plt.figure(figsize=(10, 8 * rows))
    fig.tight_layout()

    if scaler is not None:
        centroid_history_orig = [scaler.inverse_transform(centroids) for centroids in centroid_history]
    else:
        centroid_history_orig = centroid_history

    num_clusters = centroid_history[0].shape[0]
    cluster_colors = plt.cm.viridis(np.linspace(0, 1, num_clusters))

    for i in range(min(iterations, 8)):
        ax = fig.add_subplot(rows, 3, i + 1, projection='3d')

        if i < len(labels_history):
            for cluster_idx in range(num_clusters):
                mask = labels_history[i] == cluster_idx
                ax.scatter(
                    df.loc[mask, 'A'],
                    df.loc[mask, 'A_Coef'],
                    df.loc[mask, 'LKG'],
                    color=cluster_colors[cluster_idx],
                    s=80,
                    alpha=0.6,
                    label=f'Cluster {cluster_idx + 1}'
                )

        if i < len(centroid_history_orig):
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
        ax.set_xlabel('A', fontsize=10)  # Updated label
        ax.set_ylabel('A_Coef', fontsize=10)  # Updated label
        ax.set_zlabel('LKG', fontsize=10)  # Updated label
        if i == 0:
            ax.legend(fontsize=8, loc='upper right')

    if iterations > 0:
        ax = fig.add_subplot(rows, 3, min(iterations, 9), projection='3d')

        if len(labels_history) > 0:
            for cluster_idx in range(num_clusters):
                mask = labels_history[-1] == cluster_idx
                ax.scatter(
                    df.loc[mask, 'A'],
                    df.loc[mask, 'A_Coef'],
                    df.loc[mask, 'LKG'],
                    color=cluster_colors[cluster_idx],
                    s=80,
                    alpha=0.6,
                    label=f'Cluster {cluster_idx + 1}'
                )

        if len(centroid_history_orig) > 0:
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
        ax.set_xlabel('A', fontsize=10)  # Updated label
        ax.set_ylabel('A_Coef', fontsize=10)  # Updated label
        ax.set_zlabel('LKG', fontsize=10)  # Updated label
        ax.legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    plt.show()

def create_kmeans_animation(X_scaled, df, centroid_history, labels_history, scaler=None):
    """
    Create an animation of K-means iterations - updated features
    """
    if scaler is not None:
        centroid_history_orig = [scaler.inverse_transform(centroids) for centroids in centroid_history]
    else:
        centroid_history_orig = centroid_history

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Setup plot limits - updated features
    ax.set_xlim([df['A'].min() - 0.5, df['A'].max() + 0.5])
    ax.set_ylim([df['A_Coef'].min() - 0.5, df['A_Coef'].max() + 0.5])
    ax.set_zlim([df['LKG'].min() - 0.5, df['LKG'].max() + 0.5])

    ax.set_xlabel('A')  # Updated label
    ax.set_ylabel('A_Coef')  # Updated label
    ax.set_zlabel('LKG')  # Updated label

    num_clusters = centroid_history[0].shape[0]
    cluster_colors = plt.cm.viridis(np.linspace(0, 1, num_clusters))

    scatter_plots = []
    for cluster_idx in range(num_clusters):
        mask = labels_history[0] == cluster_idx
        scatter = ax.scatter(
            df.loc[mask, 'A'],
            df.loc[mask, 'A_Coef'],
            df.loc[mask, 'LKG'],
            color=cluster_colors[cluster_idx],
            s=80,
            alpha=0.6,
            label=f'Cluster {cluster_idx + 1}'
        )
        scatter_plots.append(scatter)

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
        for cluster_idx in range(num_clusters):
            mask = labels_history[frame] == cluster_idx
            if np.any(mask):
                scatter_plots[cluster_idx]._offsets3d = (
                    df.loc[mask, 'A'],
                    df.loc[mask, 'A_Coef'],
                    df.loc[mask, 'LKG']
                )
            else:
                scatter_plots[cluster_idx]._offsets3d = ([], [], [])
            centroid_plots[cluster_idx]._offsets3d = (
                [centroid_history_orig[frame][cluster_idx, 0]],
                [centroid_history_orig[frame][cluster_idx, 1]],
                [centroid_history_orig[frame][cluster_idx, 2]]
            )
        return scatter_plots + centroid_plots + [title]

    anim = FuncAnimation(fig, update, frames=len(centroid_history),
                         interval=1000, blit=False)

    plt.tight_layout()
    return anim