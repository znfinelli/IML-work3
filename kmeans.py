import numpy as np
import pandas as pd
from typing import Optional


class KMeans:
    """
    K-Means clustering algorithm.
    
    """
    
    def __init__(self, n_clusters: int, max_iters: int = 300, tol: float = 1e-4, random_state: Optional[int] = None):
        """
        Initialize K-Means clustering.
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters to form
        max_iters : int, default=300
            Maximum number of iterations
        tol : float, default=1e-4
            Tolerance for convergence (change in centroids)
        random_state : int, optional
            Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None
        
    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """
        Initialize centroids randomly.
        
        This method can be overridden for different initialization strategies
        (e.g., K-means++).
        
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
        
        # Randomly select n_clusters data points as initial centroids
        indices = np.random.choice(n_samples, size=self.n_clusters, replace=False)
        centroids = X[indices].copy()
        
        return centroids
    
    def _compute_distances(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Compute squared Euclidean distances from each point to each centroid.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data of shape (n_samples, n_features)
        centroids : np.ndarray
            Centroids of shape (n_clusters, n_features)
            
        Returns:
        --------
        np.ndarray
            Distance matrix of shape (n_samples, n_clusters)
        """
        # Compute squared Euclidean distances using broadcasting
        # (X - centroids)^2 for each combination
        distances = np.sum((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
        return distances
    
    def _assign_clusters(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Assign each data point to the nearest centroid.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data of shape (n_samples, n_features)
        centroids : np.ndarray
            Centroids of shape (n_clusters, n_features)
            
        Returns:
        --------
        np.ndarray
            Cluster assignments of shape (n_samples,)
        """
        distances = self._compute_distances(X, centroids)
        labels = np.argmin(distances, axis=1)
        return labels
    
    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Update centroids as the mean of points assigned to each cluster.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data of shape (n_samples, n_features)
        labels : np.ndarray
            Cluster assignments of shape (n_samples,)
            
        Returns:
        --------
        np.ndarray
            Updated centroids of shape (n_clusters, n_features)
        """
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        
        for k in range(self.n_clusters):
            # Get all points assigned to cluster k
            cluster_points = X[labels == k]
            
            if len(cluster_points) > 0:
                # Update centroid as the mean of cluster points
                centroids[k] = np.mean(cluster_points, axis=0)
            else:
                # If cluster is empty, keep the previous centroid
                # (or reinitialize randomly if this is the first iteration)
                centroids[k] = X[np.random.choice(X.shape[0])]
        
        return centroids
    
    def _compute_inertia(self, X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
        """
        Compute the within-cluster sum of squared distances (inertia).
        
        Parameters:
        -----------
        X : np.ndarray
            Input data of shape (n_samples, n_features)
        labels : np.ndarray
            Cluster assignments of shape (n_samples,)
        centroids : np.ndarray
            Centroids of shape (n_clusters, n_features)
            
        Returns:
        --------
        float
            Inertia value
        """
        inertia = 0.0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                # Sum of squared distances from points to their centroid
                inertia += np.sum((cluster_points - centroids[k]) ** 2)
        return inertia
    
    def fit(self, X):
        """
        Fit the K-Means model to the data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data. Can be numpy array or pandas DataFrame.
            
        Returns:
        --------
        self
            Returns self for method chaining
        """
        # Convert to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.array(X)
        
        # Initialize centroids
        self.centroids = self._initialize_centroids(X)
        
        # Main iteration loop
        for iteration in range(self.max_iters):
            # Assign points to nearest centroids
            labels = self._assign_clusters(X, self.centroids)
            
            # Update centroids
            new_centroids = self._update_centroids(X, labels)
            
            # Check for convergence
            centroid_shift = np.sum((new_centroids - self.centroids) ** 2)
            
            if centroid_shift < self.tol:
                break
            
            self.centroids = new_centroids
        
        # Store final labels and compute inertia
        self.labels_ = self._assign_clusters(X, self.centroids)
        self.inertia_ = self._compute_inertia(X, self.labels_, self.centroids)
        
        return self
    
    def predict(self, X):
        """
        Predict the closest cluster for each sample in X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            New data to predict. Can be numpy array or pandas DataFrame.
            
        Returns:
        --------
        np.ndarray
            Cluster assignments of shape (n_samples,)
        """
        if self.centroids is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        # Convert to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.array(X)
        
        # Assign points to nearest centroids
        labels = self._assign_clusters(X, self.centroids)
        
        return labels
    
    def fit_predict(self, X):
        """
        Fit the model and return cluster assignments.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data. Can be numpy array or pandas DataFrame.
            
        Returns:
        --------
        np.ndarray
            Cluster assignments of shape (n_samples,)
        """
        self.fit(X)
        return self.labels_


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_blobs
    
    # Generate sample data
    X, y_true = make_blobs(n_samples=300, centers=4, n_features=2, random_state=42)
    
    # Create and fit K-Means model
    kmeans = KMeans(n_clusters=4, random_state=42)
    labels = kmeans.fit_predict(X)
    
    print(f"Number of clusters: {kmeans.n_clusters}")
    print(f"Inertia: {kmeans.inertia_:.2f}")
    print(f"Centroids shape: {kmeans.centroids.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Unique labels: {np.unique(labels)}")

