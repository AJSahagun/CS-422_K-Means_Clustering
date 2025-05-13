from src.ver_4.preprocessing import load_and_preprocess_data
from src.custom_kmeans import SeedKMeans
from src.ver_4.visualization import (
    plot_clustering_analysis, calculate_wcss, find_elbow_point,
    plot_kmeans_iterations, create_kmeans_animation
)
import pandas as pd
import matplotlib.pyplot as plt


def main():
    # Load and preprocess data - updated to new dataset path
    X, y, scaler = load_and_preprocess_data('data/Seed_Data.csv')
    df = pd.read_csv('data/Seed_Data.csv')

    # Calculate WCSS values for elbow method
    wcss_values = calculate_wcss(X, SeedKMeans, max_k=9)

    # Train final model with optimal K
    optimal_k = find_elbow_point(wcss_values)
    final_model = SeedKMeans(n_clusters=optimal_k)
    final_model.fit(X)

    # Get predictions and evaluation metrics
    labels = final_model.predict(X)
    metrics = final_model.evaluate(X, y)

    # Get iteration history
    history = final_model.get_iteration_history()

    # Create clustering analysis visualization
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

    # Show the K-means iteration steps
    print(f"\nK-means converged after {history['iterations']} iterations")
    plot_kmeans_iterations(
        X_scaled=X,
        df=df,
        centroid_history=history['centroid_history'],
        labels_history=history['labels_history'],
        iterations=history['iterations'],
        scaler=scaler
    )

    # Create and show the K-means animation
    anim = create_kmeans_animation(
        X_scaled=X,
        df=df,
        centroid_history=history['centroid_history'],
        labels_history=history['labels_history'],
        scaler=scaler
    )

    # Save animation (optional)
    # anim.save('kmeans_iterations.gif', writer='pillow', fps=1)

    # Display the animation
    plt.show()

    # Print metrics
    print("\nModel Evaluation Metrics")
    print(f"Optimal K: {optimal_k}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1-Score: {metrics['f1_score']:.3f}")


main()