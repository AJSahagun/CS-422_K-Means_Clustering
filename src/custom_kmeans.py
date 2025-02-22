import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


class KMeans:
    def __init__(self, k=3, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.labels_ = None
        self.X_fit = None  # Store training data for inertia calculation

    def fit(self, X):
        # Store X for inertia calculation
        self.X_fit = X
        
        # initialize centroids randomly from the dataset
        np.random.seed(42)
        self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]

        for _ in range(self.max_iters):
            # assign clusters based on closest centroid
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)

            # compute new centroids
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.k)])

            # check for convergence
            if np.linalg.norm(self.centroids - new_centroids) < self.tol:
                break

            self.centroids = new_centroids

        self.labels_ = labels
        return self

    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    @property
    def cluster_centers_(self):
        return self.centroids

    @property
    def inertia_(self):
        """Calculate Within-Cluster Sum of Squares (WCSS)"""
        if self.X_fit is None:
            raise ValueError("Model must be fitted before calculating inertia")
        distances = np.linalg.norm(self.X_fit[:, np.newaxis] - self.centroids, axis=2)
        return np.sum(np.min(distances, axis=1) ** 2)


class IrisKMeans:
    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        np.random.seed(random_state)
        self.model = KMeans(k=n_clusters)

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