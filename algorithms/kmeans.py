"""
Standard K-Means Algorithm Implementation.

This module provides a custom implementation of the K-Means clustering algorithm
as required by Task 1.4 of the assignment. It implements Lloyd's algorithm
(Batch K-Means) with support for Euclidean and Manhattan (L1) distances.

References
----------
[1] Celebi, M.E., Kingravi, H.A., Vela, P.A., "A Comparative Study of Efficient
    Initialization Methods for the K-Means Clustering Algorithm", 2013,
    Expert Systems with Applications, 40(1): 200-210.
[2] MacQueen, J., "Some methods for classification and analysis of multivariate
    observations", 1967, Proc. 5th Berkeley Symp. Math. Stat. Prob., pp. 281-297.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union

class KMeans:
    """
    K-Means clustering algorithm implementation (Lloyd's Algorithm).

    This class implements the standard batch K-Means algorithm[cite: 1545].
    It iteratively assigns points to the nearest centroid and updates centroids
    to the mean of the assigned points until convergence.

    Parameters
    ----------
    n_clusters : int
        The number of clusters to form as well as the number of centroids to generate.
    max_iters : int, default=300
        Maximum number of iterations of the k-means algorithm for a single run.
    tol : float, default=1e-4
        Relative tolerance with regards to inertia to declare convergence.
    metric : str, default='euclidean'
        The distance metric to use.
        - 'euclidean': Standard L2 distance (Equation 1 in [cite: 1533]).
        - 'manhattan': L1 distance, more robust to outliers.
    random_state : int, optional
        Determines random number generation for centroid initialization.
    """

    def __init__(
        self,
        n_clusters: int,
        max_iters: int = 300,
        tol: float = 1e-4,
        metric: str = 'euclidean',
        random_state: Optional[int] = None
    ):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.metric = metric
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None

        if self.metric not in ['euclidean', 'manhattan', 'kernel']:
            raise ValueError(f"Metric '{self.metric}' not supported.")

    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """
        Initialize centroids using MacQueen's second method (Random Selection).

        "The second method chooses the centers randomly from the data points."
        . This is distinct from Forgy's method.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Initial centroids of shape (n_clusters, n_features).
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples, _ = X.shape

        # Randomly select n_clusters data points as initial centroids
        indices = np.random.choice(n_samples, size=self.n_clusters, replace=False)
        centroids = X[indices].copy()

        return centroids

    def _compute_distances(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Compute distances from each point to each centroid.

        Parameters
        ----------
        X : np.ndarray
            Input data.
        centroids : np.ndarray
            Current centroids.

        Returns
        -------
        np.ndarray
            Distance matrix of shape (n_samples, n_clusters).
        """
        if self.metric == 'euclidean':
            # Squared Euclidean distance (Standard K-Means objective) [cite: 1533]
            # (X - centroids)^2
            distances = np.sum((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
        elif self.metric == 'manhattan':
            # Manhattan (L1) distance for robustness
            distances = np.sum(np.abs(X[:, np.newaxis, :] - centroids[np.newaxis, :, :]), axis=2)
        else:
            # Placeholder for subclasses (e.g., Kernel K-Means)
            distances = np.zeros((X.shape[0], self.n_clusters))

        return distances

    def _assign_clusters(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Assign each data point to the nearest centroid.

        "Each point is assigned to the nearest center." [cite: 1547]

        Parameters
        ----------
        X : np.ndarray
            Input data.
        centroids : np.ndarray
            Current centroids.

        Returns
        -------
        np.ndarray
            Index of the nearest centroid for each sample.
        """
        distances = self._compute_distances(X, centroids)
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Recalculate centroids as the mean of points assigned to them.

        "each center is recalculated as the mean of all points assigned to it."
        [cite: 1547]

        Parameters
        ----------
        X : np.ndarray
            Input data.
        labels : np.ndarray
            Current cluster assignments.

        Returns
        -------
        np.ndarray
            Updated centroids.
        """
        centroids = np.zeros((self.n_clusters, X.shape[1]))

        for k in range(self.n_clusters):
            cluster_points = X[labels == k]

            if len(cluster_points) > 0:
                centroids[k] = np.mean(cluster_points, axis=0)
            else:
                # Handle empty clusters by reinitializing to a random point
                # This prevents NaNs if a cluster becomes empty during iteration
                centroids[k] = X[np.random.choice(X.shape[0])]

        return centroids

    def _compute_inertia(self, X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
        """
        Compute the within-cluster sum of squared errors (SSE).

        This corresponds to the standard K-Means objective function:
        SSE = sum ||x_j - c_i||^2[cite: 1533].

        Parameters
        ----------
        X : np.ndarray
            Input data.
        labels : np.ndarray
            Cluster assignments.
        centroids : np.ndarray
            Centroids.

        Returns
        -------
        float
            Total inertia (SSE).
        """
        inertia = 0.0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                if self.metric == 'euclidean':
                    inertia += np.sum((cluster_points - centroids[k]) ** 2)
                elif self.metric == 'manhattan':
                    inertia += np.sum(np.abs(cluster_points - centroids[k]))
        return inertia

    def fit(self, X: Union[np.ndarray, pd.DataFrame]):
        """
        Fit the K-Means model to the data using Lloyd's algorithm.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.array(X)

        self.centroids = self._initialize_centroids(X)

        for _ in range(self.max_iters):
            labels = self._assign_clusters(X, self.centroids)
            new_centroids = self._update_centroids(X, labels)

            # Convergence check: shift in centroids
            centroid_shift = np.sum((new_centroids - self.centroids) ** 2)
            self.centroids = new_centroids

            if centroid_shift < self.tol:
                break

        self.labels_ = self._assign_clusters(X, self.centroids)
        self.inertia_ = self._compute_inertia(X, self.labels_, self.centroids)

        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict the closest cluster for each sample in X.

        Parameters
        ----------
        X : array-like
            New data to predict.

        Returns
        -------
        np.ndarray
            Cluster assignments.
        """
        if self.centroids is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.array(X)

        return self._assign_clusters(X, self.centroids)

    def fit_predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Fit the model and return cluster assignments.

        Parameters
        ----------
        X : array-like
            Training data.

        Returns
        -------
        np.ndarray
            Cluster assignments.
        """
        self.fit(X)
        return self.labels_