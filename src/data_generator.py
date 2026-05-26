import numpy as np

def generate_kmeans_data(assignment_id=1):
    np.random.seed(42)
    
    
    mean1 = [2, 2]
    mean2 = [8, 3]
    mean3 = [3, 6]
    
    
    sigma_identity = [[1, 0], [0, 1]]
    sigma_2 = [[10, 0], [0, 1]]
    
    if assignment_id == 1:
        n_points = 200
        data1 = np.random.multivariate_normal(mean1, sigma_identity, n_points)
        data2 = np.random.multivariate_normal(mean2, sigma_identity, n_points)
        data3 = np.random.multivariate_normal(mean3, sigma_identity, n_points)
        
    elif assignment_id == 2:
        data1 = np.random.multivariate_normal(mean1, sigma_identity, 1200)
        data2 = np.random.multivariate_normal(mean2, sigma_identity, 200)
        data3 = np.random.multivariate_normal(mean3, sigma_identity, 1000)
        
    elif assignment_id == 3:
        n_points = 200
        data1 = np.random.multivariate_normal(mean1, sigma_identity, n_points)
        data2 = np.random.multivariate_normal(mean2, sigma_identity, n_points)
        data3 = np.random.multivariate_normal(mean3, sigma_2, n_points)
        
    else:
        raise ValueError("Assignment ID must be 1, 2, or 3")
    
    return np.vstack((data1, data2, data3))

