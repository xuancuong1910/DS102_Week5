import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_generator import generate_kmeans_data
from src.gmm import GMM

def draw_ellipse(position, covariance, ax=None, **kwargs):
   
    ax = ax or plt.gca()
    
   
    if covariance.shape == (2, 2):
        vals, vecs = np.linalg.eigh(covariance)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        
       
        width, height = 2 * 2 * np.sqrt(vals)
        ellip = Ellipse(xy=position, width=width, height=height,
                        angle=theta, **kwargs)
        ax.add_patch(ellip)

def main():
    
    print("--- Training GMM on Toy Dataset ---")
    X = generate_kmeans_data(assignment_id=3)

  
    k = 3
    model = GMM(k=k, max_iters=150)
    means, covariances = model.fit(X)
    labels = model.predict(X)


    plt.figure(figsize=(12, 8))
    

    colors = ['red', 'green', 'blue']
    for i in range(k):
        cluster_data = X[labels == i]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c=colors[i], 
                    label=f'Cluster {i+1}', alpha=0.4, edgecolors='none')
        
        
        draw_ellipse(means[i], covariances[i], alpha=0.2, color=colors[i])

    plt.scatter(means[:, 0], means[:, 1], c='black', marker='X', s=200, label='Means (Mu)')

    plt.title('Gaussian Mixture Model (EM) - Assignment 1')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal')
    plt.show()

    print("\n--- GMM Training Complete ---")
    print("Weights (Pi):", model.weights)
    print("Means (Mu):\n", model.means)

if __name__ == "__main__":
    main()