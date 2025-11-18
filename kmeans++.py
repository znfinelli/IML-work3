import numpy as np
from kmeans import KMeans

class KMeansPP(KMeans):
    """
    K-Means clustering algorithm with K-Means++ initialization.
    """
    
    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """
        Initialize centroids using K-Means++ algorithm.
        
        1. Choose one center uniformly at random from among the data points.
        2. For each data point x, compute D(x), the distance between x and the nearest center that has already been chosen.
        3. Choose one new data point at random as a new center, using a weighted probability distribution where a point x is chosen with probability proportional to D(x)^2.
        4. Repeat Steps 2 and 3 until k centers have been chosen.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data of shape (n_samples, n_features)
            
        Returns:
        --------
        np.ndarray
            Initial centroids of shape (n_clusters, n_features)
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        n_samples, n_features = X.shape
        centroids = np.zeros((self.n_clusters, n_features))
        
        # 1. Choose first centroid uniformly at random
        first_index = np.random.choice(n_samples)
        centroids[0] = X[first_index]
        
        # Keep track of chosen indices to avoid duplicates
        
        for k in range(1, self.n_clusters):
            # 2. Compute distances from each point to the nearest existing centroid
            # We only need to consider centroids 0 to k-1
            current_centroids = centroids[:k]
            
            # Compute squared Euclidean distances to all current centroids
            # Shape: (n_samples, k)
            dists = np.sum((X[:, np.newaxis, :] - current_centroids[np.newaxis, :, :]) ** 2, axis=2)
            
            # Find nearest distance for each point (min over current centroids)
            # Shape: (n_samples,)
            min_dists_sq = np.min(dists, axis=1)
            
            # 3. Choose next centroid with probability proportional to D(x)^2
            probs = min_dists_sq / np.sum(min_dists_sq)
            
            # Handle potential numerical issues if sum is 0 (all points on top of centroids)
            if np.sum(min_dists_sq) == 0:
                probs = np.ones(n_samples) / n_samples
                
            next_index = np.random.choice(n_samples, p=probs)
            centroids[k] = X[next_index]
            
        return centroids


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_blobs
    
    # Generate sample data
    X, y_true = make_blobs(n_samples=300, centers=4, n_features=2, random_state=42)
    
    print("Testing K-Means++ (Euclidean):")
    kmeans_pp = KMeansPP(n_clusters=4, metric='euclidean', random_state=42)
    labels_pp = kmeans_pp.fit_predict(X)
    print(f"Inertia: {kmeans_pp.inertia_:.2f}")
    
    print("\nTesting K-Means++ (Manhattan):")
    kmeans_pp_man = KMeansPP(n_clusters=4, metric='manhattan', random_state=42)
    labels_pp_man = kmeans_pp_man.fit_predict(X)
    print(f"Inertia: {kmeans_pp_man.inertia_:.2f}")
