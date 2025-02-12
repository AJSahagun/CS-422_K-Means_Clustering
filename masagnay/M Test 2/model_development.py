# model_development.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def preprocess_data(filepath):
    """Load and preprocess data"""
    df = pd.read_csv(filepath)
    X = df[['PetalLengthCm', 'PetalWidthCm']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler, df

def kmeans_clustering(X_scaled, df, k=3):
    """Perform K-means clustering with flexible K and add cluster labels to DataFrame"""
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    df['Cluster'] = kmeans.labels_  # Add cluster labels to DataFrame
    return kmeans, df

# Example usage
if __name__ == "__main__":
    X_scaled, scaler, df = preprocess_data('iris_data.csv')
    model = kmeans_clustering(X_scaled, k=3)
    df['Cluster'] = model.labels_