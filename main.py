from src.ver_1.preprocessing import load_and_preprocess_data
from src.custom_kmeans import IrisKMeans
from src.ver_1.visualization import plot_clustering_analysis, calculate_wcss, find_elbow_point
import pandas as pd

def main():
    # Load and preprocess data
    X, y, scaler = load_and_preprocess_data('data/ml_group-4_iris_dataset.csv')
    df = pd.read_csv('data/ml_group-4_iris_dataset.csv')

    # Calculate WCSS values for elbow method
    wcss_values = calculate_wcss(X, IrisKMeans, max_k=9)

    # Train final model with optimal K
    optimal_k = find_elbow_point(wcss_values)
    final_model = IrisKMeans(n_clusters=optimal_k)
    final_model.fit(X)

    # Get predictions and evaluation metrics
    labels = final_model.predict(X)
    metrics = final_model.evaluate(X, y)

    # Create visualization
    plot_clustering_analysis(
        X_scaled=X,
        df=df,
        cluster_labels=labels,
        true_labels=y,
        cluster_centers=final_model.model.cluster_centers_,
        wcss_values=wcss_values,
        conf_matrix=metrics['confusion_matrix'],
        scaler=scaler
    )

    # Print metrics
    print("\nModel Evaluation Metrics")
    print(f"Optimal K: {optimal_k}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1-Score: {metrics['f1_score']:.3f}")

if __name__ == "__main__":
    main()