import sys
import os
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_generator import generate_kmeans_data
from src.kmeans import KMeans

def main():
    print("--- Generating non-spherical data for Assignment 3 ---")
    X = generate_kmeans_data(assignment_id=3)

    k = 3
    model = KMeans(k=k, max_iters=100)
    centroids, labels = model.fit(X)

    plt.figure(figsize=(12, 7))
    colors = ['r', 'g', 'b']
    for i in range(k):
        cluster_data = X[labels == i]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c=colors[i], 
                    label=f'Cluster {i+1}', alpha=0.5)
    
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=200, label='Centroids')
    
    plt.title('K-means Clustering - Assignment 3 (Non-spherical Clusters)')
    plt.xlabel('X (Variance = 10 for Cluster 3)')
    plt.ylabel('Y (Variance = 1 for Cluster 3)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal') 
    plt.show()

if __name__ == "__main__":
    main()