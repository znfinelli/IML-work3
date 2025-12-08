"""
Intelligent Kernel K-Means Implementation.

This module implements the Kernel K-Means algorithm, an improved version of
K-Means that detects non-linear clusters by mapping data into a higher-dimensional
feature space using the Kernel Trick.

It features the "Intelligent" initialization strategy proposed by Handhayani &
Hiryanto (2015), which deterministically selects initial centroids based on
their distance from the global Center of Mass (CoM) in feature space.

References
----------
[1] Handhayani, T., & Hiryanto, L., "Intelligent Kernel K-Means for Clustering
    Gene Expression", 2015, Procedia Computer Science, 59, 171-177.
"""

import numpy as np

# Handle both relative and absolute imports for flexibility
try:
    from .kmeans import KMeans
except ImportError:
    from kmeans import KMeans


class KernelKMeans(KMeans):
    """
    Intelligent Kernel K-Means Clustering.

    Parameters
    ----------
    n_clusters : int
        Number of clusters to form.
    max_iters : int, default=300
        Maximum number of iterations.
    tol : float, default=1e-4
        Convergence tolerance.
    kernel : str, default='rbf'
        The kernel function to use.
        - 'linear': <x, y> (Equivalent to standard K-Means).
        - 'rbf': exp(-gamma * ||x - y||^2) (Captures non-linear shapes).
    gamma : float, optional
        Kernel coefficient for 'rbf'. If None, defaults to 1 / n_features.
    random_state : int, optional
        Seed for reproducibility (though initialization is largely deterministic).
    """

    def __init__(
            self,
            n_clusters: int,
            max_iters: int = 300,
            tol: float = 1e-4,
            kernel: str = 'rbf',
            gamma: float = None,
            random_state: int = None
    ):
        # We ignore 'metric' since Kernel K-Means uses implicit feature space distance
        super().__init__(
            n_clusters,
            max_iters,
            tol,
            metric='kernel',
            random_state=random_state
        )
        self.kernel_name = kernel
        self.gamma = gamma
        self.K_matrix = None  # The Kernel Matrix (N x N)
        self.center_indices = None  # Indices of points acting as centers

    def _compute_kernel_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Computes the Kernel Matrix K where K[i, j] = kernel(x_i, x_j).

        Parameters
        ----------
        X : np.ndarray
            Input data.

        Returns
        -------
        np.ndarray
            The N x N kernel matrix (float32).
        """
        X = X.astype(np.float32)

        if self.kernel_name == 'linear':
            # Eq (1) in [1]: K = X.X^T
            return np.dot(X, X.T)

        elif self.kernel_name == 'rbf':
            if self.gamma is None:
                self.gamma = 1.0 / X.shape[1]

            # Optimization: Use (a-b)^2 = a^2 + b^2 - 2ab
            # This avoids creating a massive (N, N, F) tensor
            sq_norms = np.sum(X ** 2, axis=1).reshape(-1, 1)

            # This dot product can be memory intensive (N^2 floats)
            # For 48k rows (Adult dataset), this is ~9 GB RAM.
            dot_product = np.dot(X, X.T)

            dist_sq = sq_norms + sq_norms.T - 2 * dot_product

            # Clean up immediately to free memory
            del dot_product

            # Numerical stability: Clip negative zeros
            dist_sq = np.maximum(dist_sq, 0)
            return np.exp(-self.gamma * dist_sq)

        else:
            raise ValueError(f"Unknown kernel: {self.kernel_name}")

    def _get_distance_to_mean(self, indices=None) -> np.ndarray:
        """
        Helper to calculate squared distance of every point to a centroid.

        If `indices` is None, calculates distance to the global Center of Mass.
        Uses the expansion of ||phi(x) - Mean||^2.

        References
        ----------
        [1] Eq (3): Distance to Center of Mass.
        """
        N = self.K_matrix.shape[0]
        diag_K = np.diag(self.K_matrix)  # K(x, x)

        if indices is None:
            # Distance to Global Mean (CoM)
            # Term 2: - 2/N * sum(K(x, x_i)) for all x_i
            term2 = -2.0 * np.mean(self.K_matrix, axis=1)
            # Term 3 is constant for ranking purposes, so ignored here.
            return diag_K + term2
        else:
            # Distance to the mean of a specific subset (Cluster)
            L = len(indices)
            if L == 0:
                return np.full(N, np.inf)

            # Extract relevant columns for the sum
            subset_K = self.K_matrix[:, indices]
            term2 = -2.0 / L * np.sum(subset_K, axis=1)

            return diag_K + term2

    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """
        Initializes cluster centers using the 'Intelligent' strategy.

        Strategy from Handhayani & Hiryanto [1]:
        1. Compute Center of Mass (CoM) of the entire dataset.
        2. C1 = Object farthest from CoM (Eq 3).
        3. C2 = Object farthest from C1 (Eq 4).
        4. For K > 2, we extend this using MaxMin strategy in feature space.

        Returns
        -------
        np.ndarray
            Array of indices representing the initial centers.
        """
        n_samples = X.shape[0]

        # 1. Find Global Center of Mass (CoM) distances
        dist_to_global_mean = self._get_distance_to_mean(indices=None)

        # 2. C1 is the object farthest from CoM
        c1_idx = np.argmax(dist_to_global_mean)
        center_indices = [c1_idx]

        # 3. Select remaining centers
        diag_K = np.diag(self.K_matrix)

        for _ in range(1, self.n_clusters):
            # Track min distance of every point to the SET of already chosen centers
            min_dists = np.full(n_samples, np.inf)

            for c_idx in center_indices:
                # Calculate distance from every point 'i' to center point 'c_idx'
                # Eq (4) in [1]: ||phi(x) - phi(z)||^2 = K(x,x) + K(z,z) - 2K(x,z)
                dists = diag_K + self.K_matrix[c_idx, c_idx] - 2 * self.K_matrix[:, c_idx]

                # Keep the distance to the NEAREST center found so far
                min_dists = np.minimum(min_dists, dists)

            # Select the point FARTHEST from its nearest center (MaxMin)
            next_c = np.argmax(min_dists)
            center_indices.append(next_c)

        return np.array(center_indices)

    def fit(self, X):
        """
        Compute Kernel K-Means clustering.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Input data.

        Returns
        -------
        self
        """
        X_data = X.values if hasattr(X, 'values') else X

        # 1. Precompute Kernel Matrix (Computationally intensive)
        self.K_matrix = self._compute_kernel_matrix(X_data)
        n_samples = X_data.shape[0]

        # 2. Initialize Centers (Indices)
        self.center_indices = self._initialize_centroids(X_data)

        # Initial assignment
        diag_K = np.diag(self.K_matrix)
        dists = np.zeros((n_samples, self.n_clusters))

        for k in range(self.n_clusters):
            c_idx = self.center_indices[k]
            # Eq (4): distance between two single points in feature space
            dists[:, k] = diag_K + self.K_matrix[c_idx, c_idx] - 2 * self.K_matrix[:, c_idx]

        self.labels_ = np.argmin(dists, axis=1)

        # 3. Iterative Update
        for iteration in range(self.max_iters):
            labels_prev = self.labels_.copy()
            dists = np.zeros((n_samples, self.n_clusters))

            for k in range(self.n_clusters):
                # Get indices of points currently in cluster k
                mask = (self.labels_ == k)
                Nk = np.sum(mask)

                if Nk == 0:
                    dists[:, k] = np.inf
                    continue

                # Calculate Squared Distance in Feature Space
                # Formula derived from Eq (3) in [1] for a cluster centroid mu_k:
                # ||phi(x) - mu_k||^2 = K(xx) - 2/Nk * sum(K(xy)) + 1/Nk^2 * sum(K(yz))

                # Term 2: Interaction between point x and cluster k members
                term2 = -2.0 / Nk * np.sum(self.K_matrix[:, mask], axis=1)

                # Term 3: Interaction within cluster k (Constant for the cluster)
                term3 = 1.0 / (Nk ** 2) * np.sum(self.K_matrix[np.ix_(mask, mask)])

                dists[:, k] = diag_K + term2 + term3

            # Assign new labels
            self.labels_ = np.argmin(dists, axis=1)

            if np.array_equal(self.labels_, labels_prev):
                break

        # Calculate final inertia (sum of squared distances to closest centroid)
        min_dists = np.min(dists, axis=1)
        self.inertia_ = np.sum(min_dists)

        return self

    def fit_predict(self, X):
        """
        Compute cluster centers and predict cluster index for each sample.

        Parameters
        ----------
        X : np.ndarray
            Input data.

        Returns
        -------
        labels : np.ndarray
            Index of the cluster each sample belongs to.
        """
        self.fit(X)
        return self.labels_