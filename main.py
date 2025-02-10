from preprocessing import load_and_preprocess_data
from kmeans_model import IrisKMeans
from visualization import plot_elbow_method, plot_clusters, plot_confusion_matrix
import numpy as np


def main():
    # Load and preprocess data
    X, y, scaler = load_and_preprocess_data('ml_group-4_iris_dataset.csv')

    # Find optimal K using elbow method
    k_values = range(1, 10)
    wcss_values = []

    for k in k_values:
        kmeans = IrisKMeans(n_clusters=k)
        kmeans.fit(X)
        wcss_values.append(kmeans.get_wcss())

    # Plot elbow method
    plot_elbow_method(k_values, wcss_values)

    # Train final model with optimal K (3 for Iris dataset)
    optimal_k = 3
    final_model = IrisKMeans(n_clusters=optimal_k)
    final_model.fit(X)

    # Plot clusters
    labels = final_model.predict(X)
    plot_clusters(X, labels, final_model.model.cluster_centers_)

    # Evaluate model
    metrics = final_model.evaluate(X, y)

    # Print metrics
    print("\nModel Evaluation Metrics:")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1-Score: {metrics['f1_score']:.3f}")

    # Plot confusion matrix
    unique_labels = np.unique(y)
    plot_confusion_matrix(metrics['confusion_matrix'], unique_labels)


if __name__ == "__main__":
    main()