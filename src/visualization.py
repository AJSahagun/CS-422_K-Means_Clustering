import matplotlib.pyplot as plt
import seaborn as sns


def plot_elbow_method(k_values, wcss_values):
    """
    Plot the elbow method graph
    """
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, wcss_values, 'bo-')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.title('Elbow Method for Optimal K')
    plt.grid(True)
    plt.show()


def plot_clusters(X, labels, centers=None):
    """
    Plot the clusters and their centers
    """
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')

    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, linewidth=3, label='Centroids')

    plt.xlabel('Petal Length (cm)')
    plt.ylabel('Petal Width (cm)')
    plt.title('Iris Clusters')
    plt.colorbar(scatter)
    plt.legend()
    plt.show()


def plot_confusion_matrix(conf_matrix, labels):
    """
    Plot confusion matrix
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()