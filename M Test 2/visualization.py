# visualization.py
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def calculate_wcss(X_scaled, max_k=10):
    """Calculate Within-Cluster-Sum-of-Squares for elbow method"""
    wcss = []
    for k in range(1, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
    return wcss

def plot_results(wcss, df, cluster_centers):
    """Create all required visualizations"""
    plt.figure(figsize=(15,5))
    
    # Elbow Method Plot
    plt.subplot(1, 3, 1)
    plt.plot(range(1, len(wcss)+1), wcss, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    
    # Cluster Visualization
    plt.subplot(1, 3, 2)
    plt.scatter(df['PetalLengthCm'], df['PetalWidthCm'], c=df['Cluster'], cmap='viridis')
    plt.scatter(cluster_centers[:,0], cluster_centers[:,1], s=200, c='red', marker='X')
    plt.title('K-means Clustering')
    
    # True Labels Visualization
    plt.subplot(1, 3, 3)
    plt.scatter(df['PetalLengthCm'], df['PetalWidthCm'], c=df['Species'].astype('category').cat.codes, cmap='viridis')
    plt.title('Actual Species Distribution')
    
    plt.tight_layout()
    plt.show()