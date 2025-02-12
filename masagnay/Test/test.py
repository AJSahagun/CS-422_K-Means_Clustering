import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load your data (replace with your actual file path)
df = pd.read_csv('Dataset\ml_group-4_iris_dataset.csv')  # Assuming the data is in a CSV file

# Select petal features
X = df[['PetalLengthCm', 'PetalWidthCm']]

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means with 3 clusters (since we know there are 3 Iris species)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)
df['Cluster'] = kmeans.labels_

# Visualize results
plt.figure(figsize=(12, 6))

# Plot K-Means clusters
plt.subplot(1, 2, 1)
for cluster in range(3):
    cluster_data = df[df['Cluster'] == cluster]
    plt.scatter(cluster_data['PetalLengthCm'], cluster_data['PetalWidthCm'],
                label=f'Cluster {cluster}', s=50)
    
# Plot actual species (for comparison)
plt.subplot(1, 2, 2)
for species in df['Species'].unique():
    species_data = df[df['Species'] == species]
    plt.scatter(species_data['PetalLengthCm'], species_data['PetalWidthCm'],
                label=species, s=50)

# Common formatting for both plots
for ax in plt.gcf().axes:
    ax.set(xlabel='Petal Length (cm)', ylabel='Petal Width (cm)')
    ax.grid(alpha=0.3)
    ax.legend()

plt.suptitle('Petal Dimensions: K-Means Clusters vs Actual Species')
plt.tight_layout()
plt.show()

# Show cluster characteristics vs actual species
print("\nCluster Centers (Original Scale):")
centers = scaler.inverse_transform(kmeans.cluster_centers_)
print(pd.DataFrame(centers, columns=['PetalLengthCm', 'PetalWidthCm']))

print("\nCross-tab between Cluster and Species:")
print(pd.crosstab(df['Cluster'], df['Species']))