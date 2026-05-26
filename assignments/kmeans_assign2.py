import sys
import os
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_generator import generate_kmeans_data
from src.kmeans import KMeans

def main():
    print("--- Generating unbalanced data for Assignment 2 ---")
    X = generate_kmeans_data(assignment_id=2)

    k = 3
    model = KMeans(k=k, max_iters=100)
    centroids, labels = model.fit(X)

    plt.figure(figsize=(10, 6))
    colors = ['r', 'g', 'b']
    for i in range(k):
        cluster_data = X[labels == i]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c=colors[i], 
                    label=f'Cluster {i+1} ({len(cluster_data)} points)', alpha=0.5)
    
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=200, label='Centroids')
    
    plt.title('K-means Clustering - Assignment 2 (Unbalanced Sizes)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()

    

if __name__ == "__main__":
    main()