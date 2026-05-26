import numpy as np

class KMeans:
    def __init__(self, k=3, max_iters=100, tolerance=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.centroids = None
        self.labels = None

    def initialize_centroids(self, X):
        n_samples = X.shape[0]
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_indices]
        return self.centroids

    def fit(self, X):
        self.initialize_centroids(X)
        
        for i in range(self.max_iters):
            old_centroids = self.centroids.copy()

            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            self.labels = np.argmin(distances, axis=1)

            
            new_centroids = []
            for j in range(self.k):
                cluster_points = X[self.labels == j]
                if len(cluster_points) > 0:
                    new_centroids.append(cluster_points.mean(axis=0))
                else:
                    new_centroids.append(old_centroids[j])
            
            self.centroids = np.array(new_centroids)
            diff = np.linalg.norm(self.centroids - old_centroids)
            if diff < self.tolerance:
                print(f"K-means converged at iteration {i}")
                break
                
        return self.centroids, self.labels

    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

if __name__ == "__main__":
    pass