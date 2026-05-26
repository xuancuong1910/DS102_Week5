import numpy as np

class GMM:
    def __init__(self, k=3, max_iters=100, tolerance=1e-6):
        self.k = k
        self.max_iters = max_iters
        self.tolerance = tolerance
        
        self.weights = None       
        self.means = None         
        self.covariances = None   
        self.responsibilities = None

    def _gaussian_pdf(self, X, mean, cov):
        n_features = X.shape[1]
        diff = X - mean
        
        cov += np.eye(n_features) * 1e-6
        
        det = np.linalg.det(cov)
        inv = np.linalg.inv(cov)
        
        exponent = -0.5 * np.sum(diff @ inv * diff, axis=1)
        normalization = 1.0 / np.sqrt(((2 * np.pi) ** n_features) * det)
        
        return normalization * np.exp(exponent)

    def fit(self, X):
        n_samples, n_features = X.shape
        
        self.weights = np.full(self.k, 1/self.k)
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.means = X[random_indices]
        self.covariances = np.array([np.eye(n_features) for _ in range(self.k)])
        
        log_likelihood_old = 0
        
        for i in range(self.max_iters):
            weighted_pdfs = np.zeros((n_samples, self.k))
            for k in range(self.k):
                weighted_pdfs[:, k] = self.weights[k] * self._gaussian_pdf(X, self.means[k], self.covariances[k])
            
            sum_weighted_pdfs = np.sum(weighted_pdfs, axis=1, keepdims=True)
            self.responsibilities = weighted_pdfs / (sum_weighted_pdfs + 1e-10)

            N_k = np.sum(self.responsibilities, axis=0)

            for k in range(self.k):
                self.weights[k] = N_k[k] / n_samples
                
                self.means[k] = np.sum(self.responsibilities[:, [k]] * X, axis=0) / N_k[k]
                
                diff = X - self.means[k]
                self.covariances[k] = (self.responsibilities[:, k] * diff.T) @ diff / N_k[k]

            log_likelihood_new = np.sum(np.log(sum_weighted_pdfs + 1e-10))
            if abs(log_likelihood_new - log_likelihood_old) < self.tolerance:
                print(f"GMM converged at iteration {i}")
                break
            log_likelihood_old = log_likelihood_new

        return self.means, self.covariances

    def predict(self, X):
        weighted_pdfs = np.zeros((X.shape[0], self.k))
        for k in range(self.k):
            weighted_pdfs[:, k] = self.weights[k] * self._gaussian_pdf(X, self.means[k], self.covariances[k])
        return np.argmax(weighted_pdfs, axis=1)

    def predict_proba(self, X):
        weighted_pdfs = np.zeros((X.shape[0], self.k))
        for k in range(self.k):
            weighted_pdfs[:, k] = self.weights[k] * self._gaussian_pdf(X, self.means[k], self.covariances[k])
        return weighted_pdfs / np.sum(weighted_pdfs, axis=1, keepdims=True)