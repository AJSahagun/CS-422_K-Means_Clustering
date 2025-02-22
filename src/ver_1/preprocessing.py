import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
from itertools import permutations


def load_and_preprocess_data(file_path):
    """
    Load and preprocess the Iris dataset
    """
    # Load the dataset
    df = pd.read_csv(file_path)

    # Extract features and target
    X = df[['PetalLengthCm', 'PetalWidthCm']].values
    y = df['Species'].values

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler


def create_label_mapping(true_labels, cluster_labels):
    """
    Create mapping between cluster labels (numbers) and true labels (strings)
    Handles cases where number of clusters differs from number of true labels
    """
    unique_true = np.unique(true_labels)
    unique_cluster = np.unique(cluster_labels)
    
    # If number of clusters doesn't match number of true labels
    if len(unique_cluster) != len(unique_true):
        # Create a mapping that assigns the most common true label for each cluster
        best_mapping = {}
        for cluster_label in unique_cluster:
            mask = cluster_labels == cluster_label
            true_labels_in_cluster = true_labels[mask]
            most_common_label = np.unique(true_labels_in_cluster, return_counts=True)
            best_mapping[cluster_label] = most_common_label[0][np.argmax(most_common_label[1])]
        return best_mapping
    
    # Original logic for when number of clusters matches number of true labels
    best_mapping = {}
    best_accuracy = 0

    for perm in permutations(unique_true):
        mapping = dict(zip(range(len(unique_true)), perm))
        mapped_clusters = np.array([mapping[label] for label in cluster_labels])
        acc = accuracy_score(true_labels, mapped_clusters)

        if acc > best_accuracy:
            best_accuracy = acc
            best_mapping = mapping

    return best_mapping