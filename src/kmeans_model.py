import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


class IrisKMeans:
    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state)

    def fit(self, X):
        """
        Fit the K-means model
        """
        self.model.fit(X)
        return self

    def predict(self, X):
        """
        Predict clusters
        """
        return self.model.predict(X)

    def get_wcss(self):
        """
        Get within-cluster sum of squares
        """
        return self.model.inertia_

    def evaluate(self, X, true_labels):
        """
        Evaluate the clustering model
        """
        # Get cluster predictions
        cluster_labels = self.predict(X)

        # Create mapping between cluster labels and true labels
        from src.ver_1.preprocessing import create_label_mapping
        mapping = create_label_mapping(true_labels, cluster_labels)

        # Map cluster labels (numbers) to predicted labels
        mapped_clusters = np.array([mapping[label] for label in cluster_labels])

        # Calculate metrics
        conf_matrix = confusion_matrix(true_labels, mapped_clusters)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels,
                                                                   mapped_clusters,
                                                                   average='weighted')

        return {
            'confusion_matrix': conf_matrix,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }