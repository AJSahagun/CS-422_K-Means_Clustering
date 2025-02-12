# main.py
from model_development import preprocess_data, kmeans_clustering
from model_evaluation import evaluate_clustering
from visualization import calculate_wcss, plot_results

# Member A
X_scaled, scaler, df = preprocess_data('Dataset\ml_group-4_iris_dataset.csv')
model, df = kmeans_clustering(X_scaled, df, k=3)  # Pass df to add cluster labels

# Member C
wcss = calculate_wcss(X_scaled)
cluster_centers = scaler.inverse_transform(model.cluster_centers_)
plot_results(wcss, df, cluster_centers)

# Member B
species_codes = df['Species'].astype('category').cat.codes
results = evaluate_clustering(species_codes, model.labels_)

print("Confusion Matrix:")
print(results['confusion_matrix'])
print(f"\nPrecision: {results['precision']:.3f}")
print(f"Recall: {results['recall']:.3f}")
print(f"F1-Score: {results['f1_score']:.3f}")