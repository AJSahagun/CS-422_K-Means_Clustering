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
    X = df[['SepalLengthCm', 'SepalWidthCm']].values
    y = df['Species'].values

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler


def create_label_mapping(true_labels, cluster_labels):
    """
    Create mapping between cluster labels (numbers) and true labels (strings)
    """
    unique_true = np.unique(true_labels)
    unique_cluster = np.unique(cluster_labels)

    best_mapping = {}
    best_accuracy = 0

    # Try all possible mappings
    for perm in permutations(unique_true):
        # Create a mapping from cluster numbers to true labels
        mapping = dict(zip(range(len(unique_true)), perm))
        # Map cluster labels to true labels for comparison
        mapped_clusters = np.array([mapping[label] for label in cluster_labels])
        acc = accuracy_score(true_labels, mapped_clusters)

        if acc > best_accuracy:
            best_accuracy = acc
            best_mapping = mapping

    return best_mapping