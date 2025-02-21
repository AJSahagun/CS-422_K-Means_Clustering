import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # Necessary for 3D plotting

def plot_clustering_analysis_3d(X_scaled, df, cluster_labels, true_labels, cluster_centers, wcss_values, conf_matrix, scaler=None):
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
    fig = plt.figure(figsize=(16, 12))

    # 1. Elbow Method Plot (2D)
    ax1 = fig.add_subplot(2, 2, 2)
    ax1.plot(range(1, len(wcss_values)+1), wcss_values, marker='o', linewidth=2, markersize=8)
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
    for k in range(1, max_k+1):
        kmeans = kmeans_class(n_clusters=k)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.get_wcss())
    return wcss